# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 blockscale GEMM: 256x256 tiles, eight waves on a 2x4 (M x N) grid, A and B staged through LDS.

C[M, N] = A[M, K] @ B[N, K]^T with FNUZ FP8 inputs, FP32 K128 block scales ScaleA[K/128, M] and
ScaleB[ceil(N/128), K/128], and BF16 or FP16 output. wave // 4 owns 128 rows, wave % 4 owns 64 columns. A and B
are copied cooperatively into 32 KiB of LDS each as four 64-row packets; each packet of the next K block is loaded
during the current block and written to LDS one stage later (one-stage load lead, "pf1"). Waves 4-7 trail waves
0-3 by one barrier so one group's memory phase overlaps the other's MFMAs. B is plain [N, K] or
shuffle_weight(B, layout=(16, 16)). Any M and N % 16 == 0 work: the raw-buffer range checks zero rows and columns
past M/N, and those are never stored.

Ported from the PyHIP 2x4-LDS pf1 kernel (pyhip contrib/flydsl/gemm_a8w8_blockscale_gfx942_2x4_lds_pf1.py) with
the same tile order, MFMA/dequant order and barriers; the output conversion rounds to nearest even.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

TILE = 256
TILE_K = 128
PACKET_ROWS = 64
PACKET_DWORDS = PACKET_ROWS * TILE_K // 4


def _lds_dwords(row, column):
    # XOR swizzle of the 16-byte column keeps the MFMA-pattern ds_read_b128 conflict free.
    return row * (TILE_K // 4) + ((row ^ column) & 7) * 4


def _tb_swizzle(block_id, M, N, m01=8, group_num=8):
    """Workgroup id -> (block_m, block_n) in the grouped, M01-adapted order of CK's block-to-tile map."""
    m0 = (M + TILE - 1) // TILE
    n0 = (N + TILE - 1) // TILE
    group_size = (m0 * n0 + group_num - 1) // group_num
    big_group_num = group_num - (group_size * group_num - m0 * n0)
    group_id_y = block_id // group_num
    group_id_x = block_id - group_id_y * group_num
    remap = group_id_x * group_size + group_id_y
    remap = (group_id_x > big_group_num).select(
        remap + (big_group_num - group_id_x), remap
    )
    idx_m0 = remap // n0
    idx_n0 = remap - idx_m0 * n0
    m0_mod_m01 = m0 - m0 // m01 * m01
    m01_adapt = (idx_m0 < m0 - m0_mod_m01).select(fx.Int32(m01), m0_mod_m01)
    idx_m00 = idx_m0 // m01
    local = idx_n0 + (idx_m0 - idx_m00 * m01) * n0
    block_n = local // m01_adapt
    return local - block_n * m01_adapt + idx_m00 * m01, block_n


def _s_buffer_load_f32(rsrc, offset_bytes):
    """s_buffer_load_dword hidden from LLVM (the tracked intrinsic gets conservative waits).

    Every consumer is behind an lgkmcnt(0) wait the pipeline issues anyway.
    """
    value = llvm.inline_asm(
        T.f32,
        [rsrc, fx.Int32(offset_bytes).ir_value()],
        "s_buffer_load_dword $0, $1, $2",
        "=s,s,s",
        has_side_effects=True,
    )
    return fx.Float32(value)


def _buffer_load_b128(rsrc, voffset_bytes, soffset_bytes):
    """Separate voffset/soffset preserves the packet pipeline and tail masking.

    BufferCopy combines offsets; the source PF1 schedule uses a scalar K step
    independently of the vector row offset. Inspect buffer_load_dwordx4 in ISA.
    """
    value = rocdl.raw_ptr_buffer_load(
        T.i32x4,
        rsrc,
        fx.Int32(voffset_bytes).ir_value(),
        fx.Int32(soffset_bytes).ir_value(),
        fx.Int32(0).ir_value(),
    )
    return Vec(value)


def _recast_i32(src):
    """View a global iterator as i32 words. The launcher guarantees every operand is at least 4-byte aligned."""
    return fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, src.memspace, max(src.alignment, 4)), src
    )


def compile_blockscale_gfx942_2x4_lds(N, K, preshuffle_b, kernel_name):
    """A launcher for one (N, K, B layout). The output tensor's dtype (BF16 or FP16) selects the epilogue."""
    assert N % 16 == 0 and K >= TILE_K and K % TILE_K == 0
    KB = K // TILE_K
    n_tiles = (N + TILE - 1) // TILE
    scale_rows_b = (N + 127) // 128
    b_k_bytes = TILE_K * 16 if preshuffle_b else TILE_K

    @fx.struct
    class SharedTiles:
        a: fx.Array[fx.Int32, 4 * PACKET_DWORDS, 16]
        b: fx.Array[fx.Int32, 4 * PACKET_DWORDS, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[512, 1, 1])
    def gemm_kernel(
        a: fx.Tensor,
        b: fx.Tensor,
        out: fx.Tensor,
        sa: fx.Tensor,
        sb: fx.Tensor,
        M: fx.Int32,
    ):
        pA = _recast_i32(fx.get_iter(a))
        pB = _recast_i32(fx.get_iter(b))
        pC = _recast_i32(fx.get_iter(out))
        pScaleA = fx.get_iter(sa)
        pScaleB = fx.get_iter(sb)
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        wave_m = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 256).ir_value()))
        wave_n = fx.Int32(rocdl.readfirstlane(T.i32, (wave % 4).ir_value()))
        block_m, block_n = _tb_swizzle(fx.block_idx.x, M, N)
        row_base = block_m * TILE
        col_base = block_n * TILE
        rows_left = M - row_base
        rows = (rows_left < TILE).select(rows_left, TILE)
        cols_left = N - col_base
        cols = (cols_left < TILE).select(cols_left, TILE)

        # A, B and C descriptors start at this workgroup's tile (i64 base), so offsets stay tile-local.
        a_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                pA + fx.Int64(row_base) * (K // 4), fx.make_layout(TILE * K // 4, 1)
            ),
            max_size=False,
            num_records_bytes=rows * K,
        )
        b_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                pB + fx.Int64(col_base) * (K // 4), fx.make_layout(TILE * K // 4, 1)
            ),
            max_size=False,
            num_records_bytes=cols * K,
        )
        c_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                pC + fx.Int64(row_base) * (N // 2), fx.make_layout(TILE * N // 2, 1)
            ),
            max_size=False,
            num_records_bytes=rows * (N * 2),
        )
        scale_a_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pScaleA, fx.make_layout(M * KB, 1)),
            max_size=False,
            num_records_bytes=M * (KB * 4),
        )
        scale_b_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pScaleB, fx.make_layout(scale_rows_b * KB, 1)),
            max_size=False,
            num_records_bytes=scale_rows_b * KB * 4,
        )
        a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(a_tensor))
        b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(b_tensor))
        scale_a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale_a_tensor))
        scale_b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale_b_tensor))
        c_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(c_tensor))

        # Row offsets stay in voffset so raw-buffer range checks zero rows/columns past M/N.
        copy_row = tid // 8
        copy_col = tid % 8
        a_voffsets = [
            (copy_row + packet * PACKET_ROWS) * K + copy_col * 16
            for packet in range_constexpr(4)
        ]
        a_write = _lds_dwords(copy_row, copy_col)
        if const_expr(preshuffle_b):
            b_voffsets = [
                (wave // 2 + packet * 4) * (16 * K) + wave % 2 * 1024 + lane * 16
                for packet in range_constexpr(4)
            ]
            b_write = _lds_dwords(wave // 2 * 16 + lane % 16, wave % 2 * 4 + lane // 16)
        else:
            b_voffsets = a_voffsets
            b_write = a_write
        local_row = wave_m * 16 + lane % 16
        a_read = [
            [
                _lds_dwords(local_row + rep_m * 32, lane // 16 + half_k * 4)
                for half_k in range_constexpr(2)
            ]
            for rep_m in range_constexpr(2)
        ]
        b_read = [
            [
                _lds_dwords(
                    wave_n * 16 + rep_n * 64 + lane % 16, lane // 16 + half_k * 4
                )
                for half_k in range_constexpr(2)
            ]
            for rep_n in range_constexpr(2)
        ]
        last_row = M - 1
        scale_a_voffsets = []
        for packet in range_constexpr(4):
            for rep_m in range_constexpr(2):
                row = row_base + local_row + (packet * PACKET_ROWS + rep_m * 32)
                # Rows past M are never stored; clamping keeps the K-block soffset in bounds.
                scale_a_voffsets.append((row < last_row).select(row, last_row) * 4)
        scale_b_offsets = [col_base // 128 * (KB * 4), (col_base // 128 + 1) * (KB * 4)]
        last_scale_b = (N - 1) // 128 * (KB * 4)
        scale_b_offsets[1] = (scale_b_offsets[1] < last_scale_b).select(
            scale_b_offsets[1], last_scale_b
        )

        lds = fx.SharedAllocator().allocate(SharedTiles).peek()

        def lds_view(base, dwords):
            return fx.make_view(base + dwords, fx.make_layout(4, 1))

        def write_a(packet):
            lds_view(lds.a.ptr, a_write + packet * PACKET_DWORDS).store(
                a_stage[None, packet].load()
            )

        def write_b(packet):
            lds_view(lds.b.ptr, b_write + packet * PACKET_DWORDS).store(
                b_stage[None, packet % 2].load()
            )

        def read_a(packet, rep_m, half_k):
            return lds_view(
                lds.a.ptr, a_read[rep_m][half_k] + packet * PACKET_DWORDS
            ).load()

        def read_b(side, rep_n, half_k):
            return lds_view(
                lds.b.ptr, b_read[rep_n][half_k] + side * 2 * PACKET_DWORDS
            ).load()

        def load_scale_a(k_block, packet, rep_m):
            value = rocdl.raw_ptr_buffer_load(
                T.f32,
                scale_a_rsrc,
                fx.Int32(scale_a_voffsets[packet * 2 + rep_m]).ir_value(),
                fx.Int32(k_block * M * 4).ir_value(),
                fx.Int32(0).ir_value(),
            )
            return Vec.from_elements([fx.Float32(value)], fx.Float32)

        def barrier():
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        # a_stage: four A packets; b_stage: two B slots. b_ops column side * 4 + rep_n * 2 + half_k.
        a_stage = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Int32)
        b_stage = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Int32)
        b_ops = fx.make_rmem_tensor(fx.make_layout((4, 8), (1, 4)), fx.Int32)
        # scale_a column slot * 2 + rep_m (slot = packet % 2); scale_b column slot * 2 + side (slot = block % 2).
        scale_a = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
        scale_b = fx.make_rmem_tensor(fx.make_layout((1, 4), (1, 1)), fx.Float32)
        # accum column (packet * 2 + side) * 4 + rep_m * 2 + rep_n; partial column rep_m * 2 + rep_n.
        accum = fx.make_rmem_tensor(fx.make_layout((4, 32), (1, 4)), fx.Float32)
        partial = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Float32)
        partial_scale = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)

        def load_scale_b(slot, k_block):
            for side in range_constexpr(2):
                scale_b[None, slot * 2 + side].store(
                    Vec.from_elements(
                        [
                            _s_buffer_load_f32(
                                scale_b_rsrc, scale_b_offsets[side] + k_block * 4
                            )
                        ],
                        fx.Float32,
                    )
                )

        def run_k(slot, k_block):
            # The last block re-fetches clamped blocks so it stays inside the loop body.
            k_next = (k_block + 1 < KB).select(k_block + 1, KB - 1)
            for packet in range_constexpr(4):
                if const_expr(packet == 0):
                    write_a(3)
                else:
                    write_a(packet - 1)
                    write_b(packet - 1)
                    if const_expr(packet == 3):
                        write_b(3)
                rocdl.sched_barrier(0)
                if const_expr(packet == 0):
                    for side in range_constexpr(2):
                        for rep_n in range_constexpr(2):
                            for half_k in range_constexpr(2):
                                b_ops[None, side * 4 + rep_n * 2 + half_k].store(
                                    read_b(side, rep_n, half_k)
                                )
                a_ops = [
                    [
                        Vec(read_a(packet, rep_m, half_k)).bitcast(fx.Int64)
                        for half_k in range_constexpr(2)
                    ]
                    for rep_m in range_constexpr(2)
                ]
                rocdl.sched_barrier(0)
                for rep_m in range_constexpr(2):
                    if const_expr(packet < 3):
                        scale_a[None, (packet + 1) % 2 * 2 + rep_m].store(
                            load_scale_a(k_block, packet + 1, rep_m)
                        )
                    else:
                        scale_a[None, rep_m].store(load_scale_a(k_next, 0, rep_m))
                a_stage[None, packet].store(
                    _buffer_load_b128(a_rsrc, a_voffsets[packet], k_next * TILE_K)
                )
                if const_expr(packet < 3):
                    b_stage[None, packet % 2].store(
                        _buffer_load_b128(
                            b_rsrc, b_voffsets[packet], k_next * b_k_bytes
                        )
                    )
                if const_expr(packet == 2):
                    b_stage[None, 1].store(
                        _buffer_load_b128(b_rsrc, b_voffsets[3], k_next * b_k_bytes)
                    )
                rocdl.sched_barrier(0)
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                rocdl.s_setprio(3)
                if const_expr(packet == 0):
                    # Next block's B scales; consumed after several lgkmcnt(0) waits.
                    load_scale_b(1 - slot, k_next)
                    rocdl.sched_barrier(0)
                for side in range_constexpr(2):
                    previous_packet = (packet - 1) % 4 if side == 0 else packet
                    for pair_begin in range_constexpr(0, 4, 2):
                        rep_m = pair_begin // 2
                        for k_step in range_constexpr(4):
                            half_k, word = divmod(k_step, 2)
                            for rep_n in range_constexpr(2):
                                partial_index = pair_begin + rep_n
                                index = pair_begin * 4 + k_step * 2 + rep_n
                                b_value = Vec(
                                    b_ops[None, side * 4 + rep_n * 2 + half_k].load()
                                ).bitcast(fx.Int64)[word]
                                accumulator = (
                                    Vec.filled(4, 0.0, fx.Float32)
                                    if k_step == 0
                                    else Vec(partial[None, partial_index].load())
                                )
                                # SSA K32 MFMA allows the scalar dequant FMA to
                                # retire under the next MFMA; block fx.gemm does
                                # not express this instruction interleave.
                                partial[None, partial_index].store(
                                    Vec(
                                        rocdl.mfma_f32_16x16x32_fp8_fp8(
                                            T.f32x4,
                                            [
                                                b_value,
                                                a_ops[rep_m][half_k][word],
                                                accumulator,
                                                0,
                                                0,
                                                0,
                                            ],
                                        )
                                    )
                                )
                                # Keep MFMA -> FMA order: the FMA's partial then has two MFMAs of XDL latency cover.
                                rocdl.sched_barrier(0)
                                if const_expr(index == 0 or index == 8):
                                    scale = (
                                        Vec(
                                            scale_a[
                                                None, packet % 2 * 2 + index // 8
                                            ].load()
                                        )[0]
                                        * Vec(scale_b[None, slot * 2 + side].load())[0]
                                    )
                                    partial_scale[None, index // 8].store(
                                        Vec.from_elements([scale], fx.Float32)
                                    )
                                if const_expr(index < 8):
                                    source, source_scale = 2 + index // 4, 1
                                    column = (
                                        previous_packet * 2 + 1 - side
                                    ) * 4 + source
                                else:
                                    source, source_scale = (index - 8) // 4, 0
                                    column = (packet * 2 + side) * 4 + source
                                target = accum[None, column]
                                element = index % 4
                                target[element] = fx.fma(
                                    Vec(partial[None, source].load())[element],
                                    Vec(partial_scale[None, source_scale].load())[0],
                                    target[element],
                                )
                                rocdl.sched_barrier(0)
                rocdl.s_setprio(0)
                barrier()

        for packet in range_constexpr(4):
            a_stage[None, packet].store(
                _buffer_load_b128(a_rsrc, a_voffsets[packet], 0)
            )
        for pair_begin in range_constexpr(0, 4, 2):
            for packet in range_constexpr(pair_begin, pair_begin + 2):
                b_stage[None, packet % 2].store(
                    _buffer_load_b128(b_rsrc, b_voffsets[packet], 0)
                )
            for packet in range_constexpr(pair_begin, pair_begin + 2):
                write_b(packet)
            rocdl.sched_barrier(0)
        for rep_m in range_constexpr(2):
            scale_a[None, rep_m].store(load_scale_a(0, 0, rep_m))
        load_scale_b(0, 0)
        for packet in range_constexpr(4):
            write_a(packet)
        accum.fill(0)
        # The first stage retires these zeros as the "previous" side.
        for index in range_constexpr(4):
            partial[None, index].store(Vec.filled(4, 0.0, fx.Float32))
        for index in range_constexpr(2):
            partial_scale[None, index].store(Vec.filled(1, 0.0, fx.Float32))
        rocdl.s_waitcnt(lgkmcnt=0)
        barrier()
        # Waves 4-7 trail by one barrier so their memory phase overlaps waves 0-3 compute.
        if wave_m == 1:
            barrier()

        # Two blocks per iteration alternate the B-scale slots; odd KB peels block 0.
        if const_expr(KB % 2):
            run_k(0, fx.Int32(0))
        for pair in range(KB // 2):
            k_block = fx.Int32(pair) * 2 + KB % 2
            run_k(KB % 2, k_block)
            run_k(1 - KB % 2, k_block + 1)

        # The final side's partials 2-3 have no later MFMA shadow.
        final_scale = Vec(partial_scale[None, 1].load())[0]
        for source in range_constexpr(2, 4):
            target = accum[None, (3 * 2 + 1) * 4 + source]
            values = Vec(partial[None, source].load())
            for element in range_constexpr(4):
                target[element] = fx.fma(values[element], final_scale, target[element])
        if wave_m == 0:
            barrier()

        output_col = col_base + wave_n * 16 + lane // 16 * 4
        for packet in range_constexpr(4):
            for side in range_constexpr(2):
                for rep_m in range_constexpr(2):
                    for rep_n in range_constexpr(2):
                        column = output_col + (side * 128 + rep_n * 64)
                        row = local_row + (packet * PACKET_ROWS + rep_m * 32)
                        offset = (column < N).select(
                            row * (N * 2) + column * 2, fx.Int32(0x7FFFFFFF)
                        )
                        values = Vec(
                            accum[
                                None, (packet * 2 + side) * 4 + rep_m * 2 + rep_n
                            ].load()
                        )
                        packed = values.to(out.element_type).bitcast(fx.Int32)
                        rocdl.raw_ptr_buffer_store(
                            packed.ir_value(),
                            c_rsrc,
                            offset.ir_value(),
                            fx.Int32(0).ir_value(),
                            aux=ir.IntegerAttr.get(T.i32, 0),
                        )

    @flyc.jit
    def launch(
        a: fx.Tensor,
        b: fx.Tensor,
        out: fx.Tensor,
        sa: fx.Tensor,
        sb: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        # Keep the dequant FMAs scalar: v_pk_fma_f32 pairs would break the one-FMA-per-MFMA interleave.
        gemm_kernel(
            a,
            b,
            out,
            sa,
            sb,
            m,
            value_attrs={"llvm.passthrough": [["target-features", "-packed-fp32-ops"]]},
        ).launch(
            grid=((m + TILE - 1) // TILE * n_tiles, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    launch.compile_hints["target_features"] = "-packed-fp32-ops"
    return launch
