# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx950 FP8 qlen8/GQA16 page64 decode for underfilled grids.

Each four-wave CTA owns 64 query/head rows using 16x16 MFMA. Two sibling
CTAs cover the verification block and share the existing partial-buffer ABI.
Short contexts retain BF16 Q/P compute; long contexts use FP8 Q/P. KV storage
stays FP8. This separate schedule leaves the throughput kernel unchanged.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp, T
from aiter.ops.flydsl.kernels.pa_decode_fp8_wave import BF16_CONTEXT_LIMIT
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import LOG2E, create_llvm_ptr
from aiter.ops.flydsl.kernels.pa_decode_common import (
    cdiv,
    exp2_amdgcn_scalar,
    exp2_f32_fast,
    rcp_f32,
)


@functools.lru_cache(maxsize=None)
def compile_pa_decode_fp8_small(head_dim, num_partitions, softmax_scale):
    D = head_dim
    NP = num_partitions
    SCALE = softmax_scale if softmax_scale is not None else D**-0.5
    NKG = cdiv(D, 128)
    BKG = D // 32
    BKV_BYTES = (D + 128) * 64
    KV_BYTES = (D + 128) * 128

    @fx.struct
    class SharedStorage:
        data: fx.Array[fx.Int32, 2 * KV_BYTES // 4, 16]

    @flyc.kernel(known_block_size=(256, 1, 1))
    def pa_decode_fp8_small_kernel(
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
        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base = fx.recast_iter(fx.Int8, shared.data.ptr)
        block = fx.Int32(gpu.block_id("x"))
        num_sequences = fx.Int32(gpu.grid_dim.x) // 2
        sequence_base = (block // 16) * 8
        live_sequences = fx.min(fx.Int32(8), num_sequences - sequence_base)
        sequence = sequence_base + (block % 16) % live_sequences
        query_group = (block % 16) // live_sequences

        def _fp8_path(pmax_ptr, psum_ptr):
            tid = fx.Int32(gpu.thread_id("x"))
            wave = tid // 64
            lane = tid % 64
            col = lane % 16
            rg = lane // 16
            seq = sequence
            kv_h = fx.Int32(gpu.block_id("y"))
            part = fx.Int32(gpu.block_id("z"))
            n_kv = fx.Int32(gpu.grid_dim.y)
            row = query_group * 64 + wave * 16 + col
            qi = row // 16
            qh = kv_h * 16 + col

            def _ptr(off):
                ptr = fx.add_offset(lds_base, fx.make_int_tuple(off))
                ty = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16)
                return fx.recast_iter(ty, ptr)

            def _read(off):
                return fx.Vector(
                    fx.ptr_load(_ptr(off), result_type=fx.Vector.make_type(4, fx.Int32))
                )

            q_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            q_reg = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
            q_flat = fx.Tensor(
                fx.make_view(fx.get_iter(query_ptr), fx.make_layout(1 << 30, 1))
            )
            q_div = fx.logical_divide(q_flat, fx.make_layout(1, 1))

            def _q_load(offset):
                fx.copy(q_atom, fx.slice(q_div, (None, offset)), q_reg)
                return fx.Vector(fx.memref_load_vec(q_reg))

            def _page_buffer(ptr, page_offset, page_bytes):
                base = fx.add_offset(fx.get_iter(ptr), fx.make_int_tuple(page_offset))
                view = fx.make_view(base, fx.make_layout(page_bytes, 1))
                tensor = fx.rocdl.make_buffer_tensor(
                    view, num_records_bytes=fx.Int64(page_bytes).ir_value()
                )
                return fx.rocdl.get_buffer_rsrc(fx.get_iter(tensor))

            def _dma(src, offset, wave_offset):
                address = fx.Int32(fx.ptrtoint(lds_base)) + wave_offset
                uniform = fx.rocdl.readfirstlane(T.i32, address)
                dst = create_llvm_ptr(fx.Int32(uniform), 3)
                fx.rocdl.raw_ptr_buffer_load_async_lds(
                    src,
                    dst,
                    fx.Int32(16).ir_value(),
                    offset.ir_value(),
                    fx.Int32(0).ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=ir.IntegerAttr.get(T.i32, 2),
                )

            tables = buffer_ops.create_buffer_resource(
                tables_ptr,
                max_size=False,
                num_records_bytes=(fx.Index(gpu.grid_dim.x) // 2)
                * fx.Index(max_blocks)
                * 4,
            )
            lengths = buffer_ops.create_buffer_resource(lengths_ptr, max_size=True)
            ks = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
            vs = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
            context = fx.Int32(
                buffer_ops.buffer_load(lengths, seq, vec_width=1, is_scalar=True)
            )
            key_scale = fx.Int32(
                buffer_ops.buffer_load(ks, fx.Int32(0), vec_width=1, is_scalar=True)
            ).bitcast(fx.Float32)
            value_scale = fx.Int32(
                buffer_ops.buffer_load(vs, fx.Int32(0), vec_width=1, is_scalar=True)
            ).bitcast(fx.Float32)

            def _page(tt):
                return fx.Int64(
                    fx.Int32(
                        buffer_ops.buffer_load(
                            tables, seq * max_blocks + tt, vec_width=1, is_scalar=True
                        )
                    )
                )

            def _stage_kv(tt, buf):
                # Every 1024-byte wave chunk stays within a single physical page.
                for u in range_constexpr(D // 32):
                    wave_offset = wave * 1024 + u * 4096
                    pg = wave_offset // (D * 64)
                    page = _page(fx.min(tt * 2 + pg, cdiv(context, 64) - 1))
                    src = _page_buffer(key_ptr, (page * n_kv + kv_h) * (D * 64), D * 64)
                    _dma(
                        src,
                        tid * 16 + u * 4096 - pg * (D * 64),
                        buf * KV_BYTES + wave_offset,
                    )
                for pg in range_constexpr(2):
                    page = _page(fx.min(tt * 2 + pg, cdiv(context, 64) - 1))
                    src = _page_buffer(value_ptr, (page * n_kv + kv_h) * 8192, 8192)
                    for u in range_constexpr(2):
                        _dma(
                            src,
                            tid * 16 + u * 4096,
                            buf * KV_BYTES
                            + D * 128
                            + pg * 8192
                            + wave * 1024
                            + u * 4096,
                        )
                fx.rocdl.asyncmark()

            def _k_ops(buf, nt, kg):
                words = []
                for u in range_constexpr(2):
                    if const_expr((kg * 2 + u) * 64 < D):
                        off = (
                            buf * KV_BYTES
                            + (nt // 4) * D * 64
                            + (((kg * 2 + u) * 4 + rg) * 64 + (nt % 4) * 16 + col) * 16
                        )
                        chunk = _read(off)
                    else:
                        chunk = fx.Vector.filled(4, 0, fx.Int32)
                    words.extend([chunk[i] for i in range_constexpr(4)])
                return fx.Vector.from_elements(words, dtype=fx.Int32)

            def _v_ops(buf, vh):
                words = []
                for u in range_constexpr(2):
                    off = (
                        buf * KV_BYTES
                        + D * 128
                        + (rg // 2) * 8192
                        + (((rg % 2) * 2 + u) * 128 + vh * 16 + col) * 16
                    )
                    chunk = _read(off)
                    words.extend([chunk[i] for i in range_constexpr(4)])
                return fx.Vector.from_elements(words, dtype=fx.Int32)

            def _pack(values):
                words = []
                for g in range_constexpr(len(values) // 4):
                    i = g * 4
                    lo = fx.rocdl.cvt_pk_fp8_f32(
                        T.i32, values[i], values[i + 1], 0, False
                    )
                    words.append(
                        fx.Int32(
                            fx.rocdl.cvt_pk_fp8_f32(
                                T.i32, values[i + 2], values[i + 3], lo, True
                            )
                        )
                    )
                return fx.Vector.from_elements(words, dtype=fx.Int32)

            def _pack_p(prob):
                word = fx.Vector.filled(2, 0, fx.Int16).ir_value()
                word = fx.rocdl.cvt_scalef32_pk_fp8_f32(
                    T.vec(2, T.i16),
                    word,
                    prob[0].ir_value(),
                    prob[1].ir_value(),
                    fx.Float32(1.0 / 256.0).ir_value(),
                    False,
                )
                word = fx.rocdl.cvt_scalef32_pk_fp8_f32(
                    T.vec(2, T.i16),
                    word,
                    prob[2].ir_value(),
                    prob[3].ir_value(),
                    fx.Float32(1.0 / 256.0).ir_value(),
                    True,
                )
                return fx.Vector(word).bitcast(fx.Int32)[0]

            def _swap(a, b, bit):
                ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
                if const_expr(bit == 1):
                    pair = fx.rocdl.permlane16_swap(
                        ty, a.ir_value(), b.ir_value(), False, False
                    )
                else:
                    pair = fx.rocdl.permlane32_swap(
                        ty, a.ir_value(), b.ir_value(), False, False
                    )
                return fx.Int32(llvm.extractvalue(T.i32, pair, [0])), fx.Int32(
                    llvm.extractvalue(T.i32, pair, [1])
                )

            def _pair(value, bit):
                bits = value.bitcast(fx.Int32)
                lo, hi = _swap(bits, bits, bit)
                return lo.bitcast(fx.Float32), hi.bitcast(fx.Float32)

            def _mma(a, b, acc):
                return fx.Vector(
                    fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        T.f32x4,
                        [
                            a,
                            b,
                            acc,
                            0,
                            0,
                            0,
                            fx.Int32(0x7F7F7F7F),
                            0,
                            fx.Int32(0x7F7F7F7F),
                        ],
                    )
                )

            q_base = (
                fx.Int64(seq * 8 + qi) * stride_q_row + fx.Int64(qh) * stride_q_head
            )
            units = []
            for kg in range_constexpr(NKG):
                for u in range_constexpr(4):
                    vals = fx.Vector.filled(8, 0.0, fx.BFloat16)
                    if const_expr((kg * 2 + u // 2) * 64 < D):
                        vals = _q_load(
                            q_base + (kg * 2 + u // 2) * 64 + rg * 16 + (u % 2) * 8
                        )
                    units.append(vals)
            amax = fx.Float32(0.0)
            for u in range_constexpr(NKG * 4):
                amax = fx.maxnumf(
                    amax, fx.absf(units[u].to(fx.Float32)).reduce(ReductionOp.MAX)
                )
            for bit in (1, 2):
                lo, hi = _pair(amax, bit)
                amax = fx.maxnumf(lo, hi)
            qscale = fx.maxnumf(amax * fx.Float32(1.0 / 448.0), fx.Float32(1e-20))
            invq = fx.Float32(rcp_f32(qscale))
            q_ops = []
            for kg in range_constexpr(NKG):
                q_ops.append(
                    _pack(
                        [
                            units[kg * 4 + i // 8][i % 8].to(fx.Float32) * invq
                            for i in range_constexpr(32)
                        ]
                    )
                )
            scale = fx.Float32(SCALE * LOG2E) * key_scale * qscale
            zero = fx.Float32(0.0)
            neg_inf = fx.Float32(float("-inf"))
            num_tiles = cdiv(context, 128)
            per_part = cdiv(num_tiles, NP)
            begin = part * per_part
            end = fx.min(begin + per_part, num_tiles)
            if begin < end:
                _stage_kv(begin, fx.Int32(0))
                fx.rocdl.wait_asyncmark(0)
            gpu.barrier()
            out_zero = fx.Vector.filled(4, 0.0, fx.Float32)
            init = [neg_inf, zero, *([out_zero] * 8)]
            for tt, state in range(begin, end, 1, init=init):
                tt = fx.Int32(tt)
                buf = (tt - begin) & fx.Int32(1)
                if tt + 1 < end:
                    _stage_kv(tt + 1, buf ^ fx.Int32(1))
                scores = []
                local_max = neg_inf
                for nt in range_constexpr(8):
                    acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for kg in range_constexpr(NKG):
                        acc = _mma(_k_ops(buf, nt, kg), q_ops[kg], acc)
                    score = acc
                    if (tt + 1) * 128 > context - 7:
                        tokens = fx.Vector.from_elements(
                            [
                                tt * 128 + nt * 16 + rg * 4 + i
                                for i in range_constexpr(4)
                            ],
                            dtype=fx.Int32,
                        )
                        bound = fx.Vector.from_elements(
                            [context - 7 + qi], dtype=fx.Int32
                        ).broadcast_to(4)
                        score = (tokens < bound).select(
                            acc, fx.Vector.filled(4, float("-inf"), fx.Float32)
                        )
                    local_max = fx.maxnumf(local_max, score.reduce(ReductionOp.MAX))
                    scores.append(score)
                for bit in (1, 2):
                    lo, hi = _pair(local_max, bit)
                    local_max = fx.maxnumf(lo, hi)
                new_max = fx.maxnumf(state[0], local_max * scale)
                safe_max = arith.select(new_max > neg_inf, new_max, zero)
                max_vec = fx.Vector.from_elements(
                    [safe_max], dtype=fx.Float32
                ).broadcast_to(4)
                scale_vec = fx.Vector.from_elements(
                    [scale], dtype=fx.Float32
                ).broadcast_to(4)
                local_sum = zero
                p_words = []
                for nt in range_constexpr(8):
                    prob = fx.Vector(exp2_f32_fast(scores[nt] * scale_vec - max_vec))
                    local_sum = local_sum + prob.reduce(ReductionOp.ADD)
                    p_words.append(_pack_p(prob))
                for bit in (1, 2):
                    lo, hi = _pair(local_sum, bit)
                    local_sum = lo + hi
                corr = fx.Float32(exp2_amdgcn_scalar(state[0] - safe_max))
                new_sum = state[1] * corr + local_sum
                packed = []
                for nt_half in range_constexpr(2):
                    values = [p_words[2 * g + nt_half] for g in range_constexpr(4)]
                    for bit in (1, 2):
                        transposed = [None] * 4
                        for g in range_constexpr(4):
                            if const_expr((g & bit) == 0):
                                transposed[g], transposed[g + bit] = _swap(
                                    values[g], values[g + bit], bit
                                )
                        values = transposed
                    packed.extend(values)
                p = fx.Vector.from_elements(packed, dtype=fx.Int32)
                corr_vec = fx.Vector.from_elements(
                    [corr], dtype=fx.Float32
                ).broadcast_to(4)
                updated = []
                for vh in range_constexpr(8):
                    updated.append(_mma(_v_ops(buf, vh), p, state[2 + vh] * corr_vec))
                new_sum = fx.Float32(
                    llvm.inline_asm(
                        T.f32,
                        [fx.Float32(new_sum).ir_value()],
                        "",
                        "=v,0",
                        has_side_effects=True,
                    )
                )
                fx.rocdl.wait_asyncmark(0)
                gpu.barrier()
                results = yield [new_max, new_sum, *updated]
            denom = results[1]
            inv = (
                fx.Float32(rcp_f32(arith.select(denom > zero, denom, fx.Float32(1.0))))
                * value_scale
                * fx.Float32(1.0 / 256.0)
            )
            for vh in range_constexpr(8):
                norm = (
                    results[2 + vh]
                    * fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(4)
                ).to(fx.BFloat16)
                sub = vh * 4 + rg
                if const_expr(NP == 1):
                    out_row = output_ptr[seq * 8 + qi, qh, None]
                    fx.slice(
                        fx.logical_divide(out_row, fx.make_layout(4, 1)), (None, sub)
                    ).store(norm)
                else:
                    base = ((seq * n_kv + kv_h) * NP + part) * 128 + row
                    fx.slice(
                        fx.logical_divide(pout_ptr, fx.make_layout(4, 1)),
                        (None, base * 32 + sub),
                    ).store(norm)
            if const_expr(NP > 1):
                if rg == 0:
                    base = ((seq * n_kv + kv_h) * NP + part) * 128 + row
                    pmax_ptr[base] = results[0] * fx.Float32(1.0 / LOG2E)
                    psum_ptr[base] = denom

        def _bf16_path(pmax_ptr, psum_ptr):
            tid = fx.Int32(gpu.thread_id("x"))
            wave = tid // 64
            lane = tid % 64
            col = lane % 16
            rg = lane // 16
            seq = sequence
            kv_h = fx.Int32(gpu.block_id("y"))
            part = fx.Int32(gpu.block_id("z"))
            n_kv = fx.Int32(gpu.grid_dim.y)
            row = query_group * 64 + wave * 16 + col
            qi = row // 16
            qh = kv_h * 16 + row % 16

            def _lds_ptr(off):
                ptr = fx.add_offset(lds_base, fx.make_int_tuple(off))
                ty = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 8)
                return fx.recast_iter(ty, ptr)

            def _lds_read(off):
                return fx.Vector(
                    fx.ptr_load(
                        _lds_ptr(off), result_type=fx.Vector.make_type(2, fx.Int32)
                    )
                )

            def _loader(ptr, dtype, width, atom_op):
                atom = fx.make_copy_atom(atom_op, dtype)
                reg = fx.make_rmem_tensor(fx.make_layout(width, 1), dtype)
                flat = fx.Tensor(
                    fx.make_view(fx.get_iter(ptr), fx.make_layout(1 << 30, 1))
                )
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
                    view, num_records_bytes=fx.Int64(page_elems).ir_value()
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
                    source_offset.ir_value(),
                    fx.Int32(0).ir_value(),
                    fx.Int32(0).ir_value(),
                    aux=ir.IntegerAttr.get(T.i32, 2),  # CDNA4 non-temporal bit.
                )

            tables = buffer_ops.create_buffer_resource(
                tables_ptr,
                max_size=False,
                num_records_bytes=(fx.Index(gpu.grid_dim.x) // 2)
                * fx.Index(max_blocks)
                * 4,
            )
            lengths = buffer_ops.create_buffer_resource(lengths_ptr, max_size=True)
            context = fx.Int32(
                buffer_ops.buffer_load(lengths, seq, vec_width=1, is_scalar=True)
            )
            ks = buffer_ops.create_buffer_resource(key_scale_ptr, max_size=True)
            vs = buffer_ops.create_buffer_resource(value_scale_ptr, max_size=True)
            key_scale = fx.Int32(
                buffer_ops.buffer_load(ks, fx.Int32(0), vec_width=1, is_scalar=True)
            ).bitcast(fx.Float32)
            value_scale = fx.Int32(
                buffer_ops.buffer_load(vs, fx.Int32(0), vec_width=1, is_scalar=True)
            ).bitcast(fx.Float32)

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
                for u in range_constexpr(D // 64):
                    local = tid * 16 + u * 4096
                    _dma(k_src, local, buf * BKV_BYTES + wave * 1024 + u * 4096)
                for u in range_constexpr(2):
                    local = tid * 16 + u * 4096
                    _dma(
                        v_src, local, buf * BKV_BYTES + D * 64 + wave * 1024 + u * 4096
                    )
                fx.rocdl.asyncmark()

            def _decode(words):
                values = []
                for word in range_constexpr(2):
                    for hi in range_constexpr(2):
                        pair = fx.Vector(
                            fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                                T.vec(2, T.bf16),
                                words[word].ir_value(),
                                fx.Float32(1.0).ir_value(),
                                bool(hi),
                            )
                        )
                        values.extend([pair[0], pair[1]])
                return fx.Vector.from_elements(values, dtype=fx.BFloat16)

            def _k_ops(buf, nt, kg):
                off = (
                    buf * BKV_BYTES
                    + ((kg * 2 + rg // 2) * 64 + nt * 16 + col) * 16
                    + (rg % 2) * 8
                )
                return _decode(_lds_read(off))

            def _v_tile(buf, vh, step):
                off = (
                    buf * BKV_BYTES
                    + D * 64
                    + ((step * 2 + rg // 2) * 128 + vh * 16 + col) * 16
                    + (rg % 2) * 8
                )
                return _decode(_lds_read(off))

            def _swap(a, b, bit):
                ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
                if const_expr(bit == 1):
                    pair = fx.rocdl.permlane16_swap(
                        ty, a.ir_value(), b.ir_value(), False, False
                    )
                else:
                    pair = fx.rocdl.permlane32_swap(
                        ty, a.ir_value(), b.ir_value(), False, False
                    )
                return fx.Int32(llvm.extractvalue(T.i32, pair, [0])), fx.Int32(
                    llvm.extractvalue(T.i32, pair, [1])
                )

            def _pair(value, bit):
                bits = fx.Float32(value).bitcast(fx.Int32)
                lo, hi = _swap(bits, bits, bit)
                return lo.bitcast(fx.Float32), hi.bitcast(fx.Float32)

            def _mma(a, b, acc):
                return fx.Vector(
                    fx.rocdl.mfma_f32_16x16x32_bf16(T.f32x4, [a, b, acc, 0, 0, 0])
                )

            q_base = (
                fx.Int64(seq * 8 + qi) * stride_q_row + fx.Int64(qh) * stride_q_head
            )
            q_ops = [q_load(q_base + kg * 32 + rg * 8) for kg in range_constexpr(BKG)]
            scale = fx.Float32(SCALE * LOG2E) * key_scale
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
            out_zero = fx.Vector.filled(4, 0.0, fx.Float32)
            init = [neg_inf, zero, *([out_zero] * 8)]
            for tt, state in range(begin, end, 1, init=init):
                tt = fx.Int32(tt)
                buf = (tt - begin) & fx.Int32(1)
                if tt + 1 < end:
                    _stage_kv(tt + 1, buf ^ fx.Int32(1))
                scores = []
                local_max = neg_inf
                for nt in range_constexpr(4):
                    acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for kg in range_constexpr(BKG):
                        acc = _mma(_k_ops(buf, nt, kg), q_ops[kg], acc)
                    score = acc
                    if (tt + 1) * 64 > context - 7:
                        tokens = fx.Vector.from_elements(
                            [
                                tt * 64 + nt * 16 + rg * 4 + i
                                for i in range_constexpr(4)
                            ],
                            dtype=fx.Int32,
                        )
                        bound = fx.Vector.from_elements(
                            [context - 7 + qi], dtype=fx.Int32
                        ).broadcast_to(4)
                        score = (tokens < bound).select(
                            acc, fx.Vector.filled(4, float("-inf"), fx.Float32)
                        )
                    scores.append(score)
                    local_max = fx.maxnumf(local_max, score.reduce(ReductionOp.MAX))
                for bit in (1, 2):
                    lo, hi = _pair(local_max, bit)
                    local_max = fx.maxnumf(lo, hi)
                new_max = fx.maxnumf(state[0], local_max * scale)
                safe_max = arith.select(new_max > neg_inf, new_max, zero)
                scale_vec = fx.Vector.from_elements(
                    [scale], dtype=fx.Float32
                ).broadcast_to(4)
                max_vec = fx.Vector.from_elements(
                    [safe_max], dtype=fx.Float32
                ).broadcast_to(4)
                probs = [
                    fx.Vector(exp2_f32_fast(scores[nt] * scale_vec - max_vec))
                    for nt in range_constexpr(4)
                ]
                local_sum = zero
                for nt in range_constexpr(4):
                    local_sum = local_sum + probs[nt].reduce(ReductionOp.ADD)
                for bit in (1, 2):
                    lo, hi = _pair(local_sum, bit)
                    local_sum = lo + hi
                corr = fx.Float32(exp2_amdgcn_scalar(state[0] - safe_max))
                new_sum = state[1] * corr + local_sum
                p_words = [
                    probs[nt].to(fx.BFloat16).bitcast(fx.Int32)
                    for nt in range_constexpr(4)
                ]
                p = []
                for step in range_constexpr(2):
                    first, second = [], []
                    for word in range_constexpr(2):
                        lo, hi = _swap(
                            p_words[step * 2][word], p_words[step * 2 + 1][word], 2
                        )
                        a, b = _swap(lo, hi, 1)
                        first.append(a)
                        second.append(b)
                    p.append(
                        fx.Vector.from_elements(
                            [*first, *second], dtype=fx.Int32
                        ).bitcast(fx.BFloat16)
                    )
                corr_vec = fx.Vector.from_elements(
                    [corr], dtype=fx.Float32
                ).broadcast_to(4)
                updated = []
                for vh in range_constexpr(8):
                    out = state[2 + vh] * corr_vec
                    for step in range_constexpr(2):
                        out = _mma(_v_tile(buf, vh, step), p[step], out)
                    updated.append(out)
                new_sum = fx.Float32(
                    llvm.inline_asm(
                        T.f32,
                        [fx.Float32(new_sum).ir_value()],
                        "",
                        "=v,0",
                        has_side_effects=True,
                    )
                )
                fx.rocdl.wait_asyncmark(0)
                gpu.barrier()
                results = yield [new_max, new_sum, *updated]
            denom = results[1]
            inv = (
                fx.Float32(rcp_f32(arith.select(denom > zero, denom, fx.Float32(1.0))))
                * value_scale
            )
            for vh in range_constexpr(8):
                norm = (
                    results[2 + vh]
                    * fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(4)
                ).to(fx.BFloat16)
                sub = vh * 4 + rg
                if const_expr(NP == 1):
                    out_row = output_ptr[seq * 8 + qi, qh, None]
                    fx.slice(
                        fx.logical_divide(out_row, fx.make_layout(4, 1)), (None, sub)
                    ).store(norm)
                else:
                    base = ((seq * n_kv + kv_h) * NP + part) * 128 + row
                    fx.slice(
                        fx.logical_divide(pout_ptr, fx.make_layout(4, 1)),
                        (None, base * 32 + sub),
                    ).store(norm)
            if const_expr(NP > 1):
                if rg == 0:
                    base = ((seq * n_kv + kv_h) * NP + part) * 128 + row
                    pmax_ptr[base] = results[0] * fx.Float32(1.0 / LOG2E)
                    psum_ptr[base] = denom

        lengths_resource = buffer_ops.create_buffer_resource(lengths_ptr, max_size=True)
        context = fx.Int32(
            buffer_ops.buffer_load(
                lengths_resource, sequence, vec_width=1, is_scalar=True
            )
        )
        if context <= fx.Int32(BF16_CONTEXT_LIMIT):
            _bf16_path(pmax_ptr, psum_ptr)
        else:
            _fp8_path(pmax_ptr, psum_ptr)

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
            pa_decode_fp8_small_kernel(
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
            ).launch(grid=(num_seqs * 2, num_kv, NP), block=(256, 1, 1), stream=stream)

    return {"launch": launch, "kernel": pa_decode_fp8_small_kernel}
