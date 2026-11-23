# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# Based on tingqli/pyhip's MIT-licensed PF2 kernel at e609ace236f7.
# Adapted for AITER's packed gfx942 tensor ABI and FlyDSL 0.5.0.
"""Eight-wave gfx942 GEMM with independently scaled K128 partial products.

The 2×4 wave grid covers 256×256 or 224×256 outputs. Each workgroup uses
32 KiB of LDS for A and 32 KiB for B, organized into four packets per operand.
Global loads stage future packets in VGPRs until their LDS slots are available.
A uses ordinary FP8 rows, B uses shuffle_weight(B, (16, 16)), and FP32 scales
use the packed AITER ABI. The epilogue rounds once to BF16 or FP16.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

TILE_M = TILE_N = 256
TILE_K = 128
PACKET_ROWS = 64
PACKET_DWORDS = PACKET_ROWS * TILE_K // 4


def _lds_dwords(row, column):
    # XOR swizzle of the 16B column keeps MFMA-pattern ds_read_b128 conflict free.
    return row * (TILE_K // 4) + ((row ^ column) & 7) * 4


def compile_pf2(N, K, kernel_name, variant, fp16_output):
    split_k = variant == "pf2_split2"
    lds_out = variant == "pf2_ldsout"
    trimmed = variant == "pf2_vec224" or lds_out
    vector_store = variant in ("pf2_vec", "pf2_vec224", "pf2_vec_glc")
    # gfx942 GLC on the 16-byte output store is a separately tunable policy.
    vector_store_policy = 1 if variant == "pf2_vec_glc" else 0
    cache_b_scales = variant == "pf2_bscale" or trimmed
    # Each M fragment supplies 32 rows across the two M waves. A shortened
    # second packet gives the 224-row tile a smaller final grid wave at long K.
    reps = (2, 1, 2, 2) if trimmed else (2, 2, 2, 2)
    starts = (0, 64, 96, 160) if trimmed else (0, 64, 128, 192)
    tile_m = 224 if trimmed else TILE_M
    group_size = {
        "pf2_bscale": 4,
        "pf2_split2": 2,
        "pf2_vec224": 2,
        "pf2_ldsout": 2,
    }.get(variant, 1)
    # Issue independent output fragments in pairs. The LDS epilogue's kernel
    # interleaves the two K64 halves to shorten hot-loop operand lifetimes.
    issue_order = (
        ((0, 0), (2, 0), (0, 1), (2, 1), (1, 0), (3, 0), (1, 1), (3, 1))
        if lds_out
        else tuple((k_step, rep_n) for k_step in range(4) for rep_n in range(2))
    )
    cache_policy = 3 if lds_out else 0
    assert N % TILE_N == 0 and K >= TILE_K and K % TILE_K == 0
    KB = K // TILE_K
    groups = KB // 2 if split_k else KB
    n_tiles = (N + TILE_N - 1) // TILE_N
    scale_b_rows = (N + 127) // 128
    b_k_bytes = TILE_K * 16

    # Publication times, in stages relative to the start of the K128 group
    # which consumes the packet. A packet has one register slot until publication.
    write_a_at = (-3, -2, -1, 0)
    write_b_at = (-3, -2, -2, -1)
    lead_a = 3 if trimmed else 2
    lead_b = (3, 3, 3, 3) if trimmed else (3, 3, 2, 2)

    @fx.struct
    class SharedTiles:
        a: fx.Array[fx.Int32, 4 * PACKET_DWORDS, 16]
        b: fx.Array[fx.Int32, 4 * PACKET_DWORDS, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[512, 1, 1])
    def gemm_kernel(
        pA: fx.Pointer,
        pB: fx.Pointer,
        pC: fx.Pointer,
        pScaleA: fx.Pointer,
        pScaleB: fx.Pointer,
        M: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        wave_m = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 256).ir_value()))
        wave_n = fx.Int32(rocdl.readfirstlane(T.i32, (wave % 4).ir_value()))

        # Neighboring M tiles in a group reuse a packed B tile.
        num_m = (M + tile_m - 1) // tile_m
        group_tasks = group_size * n_tiles
        pid = fx.block_idx.x
        first_m = pid // group_tasks * group_size
        remaining_m = num_m - first_m
        group_m = (remaining_m < group_size).select(remaining_m, group_size)
        block_m = first_m + pid % group_tasks % group_m
        block_n = pid % group_tasks // group_m
        part = fx.Int32(fx.block_idx.y) if const_expr(split_k) else fx.Int32(0)
        k_base = part * groups if const_expr(split_k) else fx.Int32(0)
        row_base = block_m * tile_m
        col_base = block_n * TILE_N
        rows_left = M - row_base
        rows = (rows_left < tile_m).select(rows_left, tile_m)
        cols_left = N - col_base
        cols = (cols_left < TILE_N).select(cols_left, TILE_N)

        a_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                pA + fx.Int64(row_base) * (K // 4), fx.make_layout(TILE_M * K // 4, 1)
            ),
            max_size=False,
            num_records_bytes=rows * K,
        )
        b_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                pB + fx.Int64(col_base) * (K // 4), fx.make_layout(TILE_N * K // 4, 1)
            ),
            max_size=False,
            num_records_bytes=cols * K,
        )
        scale_a_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pScaleA, fx.make_layout(M * KB, 1)),
            max_size=False,
            num_records_bytes=M * (KB * 4),
        )
        scale_b_tensor = fx.rocdl.make_buffer_tensor(
            fx.make_view(pScaleB, fx.make_layout(scale_b_rows * KB, 1)),
            max_size=False,
            num_records_bytes=scale_b_rows * KB * 4,
        )
        a_rsrc = rocdl.get_buffer_rsrc(fx.get_iter(a_tensor))
        b_rsrc = rocdl.get_buffer_rsrc(fx.get_iter(b_tensor))

        scale_a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale_a_tensor))
        scale_b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(scale_b_tensor))
        if const_expr(not split_k):
            c_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    pC + fx.Int64(row_base) * (N // 2),
                    fx.make_layout(TILE_M * N // 2, 1),
                ),
                max_size=False,
                num_records_bytes=rows * (N * 2),
            )
            c_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(c_tensor))

        def load_f32(rsrc, voffset):
            # The raw descriptor load supplies the same hardware bounds check as A/B.
            zero = fx.Int32(0).ir_value()
            value = rocdl.raw_ptr_buffer_load(
                T.f32, rsrc, fx.Int32(voffset).ir_value(), zero, zero
            )
            return Vec.from_elements([fx.Float32(value)], fx.Float32)

        # Row offsets stay in voffset so raw-buffer range checks zero rows/columns past M/N.
        copy_row = tid // 8
        copy_col = tid % 8
        a_voffsets = [
            (copy_row + starts[p]) * K + copy_col * 16 for p in range_constexpr(4)
        ]
        a_write = _lds_dwords(copy_row, copy_col)

        def b_group(w, side, rep):
            if const_expr(trimmed):
                return w * 4 + side * 2 + rep
            if const_expr(vector_store):
                return side * 8 + w * 2 + rep
            return side * 8 + rep * 4 + w

        b_voffsets = [
            b_group(wave // 2, p // 2, p % 2) * (16 * K) + wave % 2 * 1024 + lane * 16
            for p in range_constexpr(4)
        ]
        b_write = _lds_dwords(wave // 2 * 16 + lane % 16, wave % 2 * 4 + lane // 16)
        a_read = [
            _lds_dwords(wave_m * 16 + lane % 16, lane // 16 + half_k * 4)
            for half_k in range_constexpr(2)
        ]
        b_read = [
            _lds_dwords(wave_n * 16 + lane % 16, lane // 16 + half_k * 4)
            for half_k in range_constexpr(2)
        ]
        scale_a_voffset = (row_base + wave_m * 16 + lane % 16) * 4
        scale_b_block = col_base // 128

        lds = fx.SharedAllocator().allocate(SharedTiles).peek()

        def lds_view(base, dwords):
            return fx.make_view(base + dwords, fx.make_layout(4, 1))

        def write_a(packet, value):
            lds_view(lds.a.ptr, a_write + packet * PACKET_DWORDS).store(value)

        def write_b(packet, value):
            lds_view(lds.b.ptr, b_write + packet * PACKET_DWORDS).store(value)

        def read_a(packet, rep_m, half_k):
            view = lds_view(
                lds.a.ptr, a_read[half_k] + (packet * PACKET_DWORDS + rep_m * 32 * 32)
            )
            return Vec(view.load()).bitcast(fx.Int64)

        def read_b(side, rep_n, half_k):
            view = lds_view(
                lds.b.ptr, b_read[half_k] + (side * 2 + rep_n) * PACKET_DWORDS
            )
            return Vec(view.load()).bitcast(fx.Int64)

        def load_a(k_block, packet):
            # Separate scalar K and vector lane offsets keep the 16-byte global load
            # independent of the LDS consumer. The generic copy atom combines offsets.
            return Vec(
                rocdl.raw_ptr_buffer_load(
                    T.i32x4,
                    a_rsrc,
                    a_voffsets[packet],
                    (k_base + k_block) * TILE_K,
                    fx.Int32(cache_policy),
                )
            )

        def load_b(k_block, packet):
            return Vec(
                rocdl.raw_ptr_buffer_load(
                    T.i32x4,
                    b_rsrc,
                    b_voffsets[packet],
                    (k_base + k_block) * b_k_bytes,
                    fx.Int32(cache_policy),
                )
            )

        def load_scale_a(k_block, packet, rep_m):
            offset = (
                scale_a_voffset
                + ((k_base + k_block) * M + (starts[packet] + rep_m * 32)) * 4
            )
            return load_f32(scale_a_rsrc, offset)

        def load_scale_b(k_block, side):
            row = scale_b_block + b_group(wave_n, side, 0) // 8
            row = (row < scale_b_rows).select(row, scale_b_rows - 1)
            return load_f32(scale_b_rsrc, (row * KB + k_base + k_block) * 4)

        def barrier():
            # A scheduling fence on each side keeps the half-workgroup phase offset
            # intact. The ordinary synchronous producer used by ID 13 is the control.
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)

        a_stage = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Int32)
        b_stage = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Int32)
        scale_a = fx.make_rmem_tensor(fx.make_layout((1, 8), (1, 1)), fx.Float32)
        scale_b = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)
        scale_b_next = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)
        c_acc = fx.make_rmem_tensor(fx.make_layout((4, 32), (1, 4)), fx.Float32)
        # The last pair of each stage retires under the next pair's MFMAs, across stage and block edges.
        pending = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Float32)
        pending_scale = fx.make_rmem_tensor(fx.make_layout((1, 2), (1, 1)), fx.Float32)
        last_pair = [30, 31]

        next_block = min(1, groups - 1)
        a_first = [load_a(0, packet) for packet in range_constexpr(3)]
        b_first = [load_b(0, packet) for packet in range_constexpr(4)]
        for p in range_constexpr(4):
            if const_expr(write_a_at[p] % 4 < lead_a):
                a_stage[None, p].store(load_a(0 if p == 3 else next_block, p))
            if const_expr(write_b_at[p] % 4 < lead_b[p]):
                b_stage[None, p].store(load_b(next_block, p))
        for packet in range_constexpr(3):
            for rep_m in range_constexpr(reps[packet]):
                scale_a[None, packet * 2 + rep_m].store(load_scale_a(0, packet, rep_m))
        for side in range_constexpr(0 if cache_b_scales else 2):
            scale_b[None, side].store(load_scale_b(0, side))
        for packet in range_constexpr(3):
            write_a(packet, a_first[packet])
        for packet in range_constexpr(4):
            write_b(packet, b_first[packet])
        c_acc.fill(0)
        for rep_n in range_constexpr(2):
            pending[None, rep_n].store(Vec.filled(4, 0.0, fx.Float32))
        pending_scale[None, 0].store(Vec.filled(1, 0.0, fx.Float32))
        rocdl.cdna3.s_waitcnt(lgkmcnt=0)
        barrier()
        # Waves 4-7 trail by one barrier so their memory phase overlaps waves 0-3 compute.
        if wave_m == 1:
            barrier()

        def run_block(k_block, left_lanes, right_lanes, scale_lane):
            k1 = (k_block + 1 < groups).select(k_block + 1, groups - 1)
            k2 = (k_block + 2 < groups).select(k_block + 2, groups - 1)
            b_ops = None
            pending_index = last_pair
            for stage in range_constexpr(4):
                for p in range_constexpr(4):
                    if const_expr(write_a_at[p] % 4 == stage):
                        write_a(p, a_stage[None, p].load())
                    if const_expr(write_b_at[p] % 4 == stage):
                        write_b(p, b_stage[None, p].load())

                if const_expr(stage == 0):
                    b_ops = [
                        [
                            [read_b(0, rep_n, half_k) for half_k in range_constexpr(2)]
                            for rep_n in range_constexpr(2)
                        ],
                        None,
                    ]
                a_ops = [
                    [read_a(stage, rep_m, half_k) for half_k in range_constexpr(2)]
                    for rep_m in range_constexpr(reps[stage])
                ]

                if const_expr(stage == 0):
                    for side in range_constexpr(0 if cache_b_scales else 2):
                        scale_b_next[None, side].store(load_scale_b(k1, side))
                    for rep_m in range_constexpr(reps[3]):
                        scale_a[None, 6 + rep_m].store(load_scale_a(k_block, 3, rep_m))
                blocks = (k_block, k1, k2)
                for p in range_constexpr(4):
                    if const_expr((write_a_at[p] - lead_a) % 4 == stage):
                        ahead = (stage - write_a_at[p] + lead_a) // 4
                        a_stage[None, p].store(load_a(blocks[ahead], p))
                    if const_expr((write_b_at[p] - lead_b[p]) % 4 == stage):
                        ahead = (stage - write_b_at[p] + lead_b[p]) // 4
                        b_stage[None, p].store(load_b(blocks[ahead], p))
                if const_expr(stage > 0):
                    for rep_m in range_constexpr(reps[stage - 1]):
                        scale_a[None, (stage - 1) * 2 + rep_m].store(
                            load_scale_a(k1, stage - 1, rep_m)
                        )

                rocdl.cdna3.s_waitcnt(lgkmcnt=0)
                barrier()
                if const_expr(stage == 0):
                    b_ops[1] = [
                        [read_b(1, rep_n, half_k) for half_k in range_constexpr(2)]
                        for rep_n in range_constexpr(2)
                    ]
                for side_index in range_constexpr(2):
                    if const_expr(cache_b_scales):
                        bank = (
                            left_lanes
                            if const_expr(side_index == 0 or trimmed)
                            else right_lanes
                        )
                        # Each lane holds one K group's weight scale. One readlane
                        # broadcasts that group's multiplier to the wave.
                        side_scale = fx.Float32(rocdl.readlane(T.f32, bank, scale_lane))
                    else:
                        side_scale = Vec(scale_b[None, side_index].load())[0]
                    for pair in range_constexpr(reps[stage]):
                        if const_expr(lds_out):
                            rep_m, side = divmod(side_index * reps[stage] + pair, 2)
                        else:
                            rep_m, side = pair, side_index
                        pair_scale = (
                            Vec(scale_a[None, stage * 2 + rep_m].load())[0] * side_scale
                        )
                        partials = [
                            Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(2)
                        ]
                        for slot in range_constexpr(8):
                            k_step, rep_n = issue_order[slot]
                            half_k, word = divmod(k_step, 2)
                            # The block MMA API does not expose individual K32 issues.
                            # Expose one K32 MFMA to the scheduler at a time;
                            # interleave it with retiring the preceding FP32 fragment.
                            partials[rep_n] = Vec(
                                rocdl.mfma_f32_16x16x32_fp8_fp8(
                                    T.f32x4,
                                    [
                                        b_ops[side][rep_n][half_k][word],
                                        a_ops[rep_m][half_k][word],
                                        partials[rep_n],
                                        0,
                                        0,
                                        0,
                                    ],
                                )
                            )
                            # Retire one element of the previous pair under each MFMA.
                            old_n, element = divmod(slot, 4)
                            old_values = Vec(pending[None, old_n].load())
                            old_scale = Vec(pending_scale[None, 0].load())[0]
                            target = c_acc[None, pending_index[old_n]]
                            target[element] = fx.fma(
                                old_values[element], old_scale, target[element]
                            )
                            rocdl.sched_barrier(0)
                        for rep_n in range_constexpr(2):
                            pending[None, rep_n].store(partials[rep_n])
                        pending_scale[None, 0].store(
                            Vec.from_elements([pair_scale], fx.Float32)
                        )
                        pending_index = [
                            ((stage * 2 + side) * 2 + rep_m) * 2 + rep_n
                            for rep_n in range_constexpr(2)
                        ]
                barrier()
            for side in range_constexpr(0 if cache_b_scales else 2):
                scale_b[None, side].store(scale_b_next[None, side].load())

        if const_expr(cache_b_scales):
            for page in range((KB + 63) // 64):
                first = fx.Int32(page) * 64
                stop = (first + 64 < KB).select(first + 64, fx.Int32(KB))
                index = (first + lane < KB).select(first + lane, fx.Int32(KB - 1))
                left = Vec(load_scale_b(index, 0))[0]
                right = left if const_expr(trimmed) else Vec(load_scale_b(index, 1))[0]
                for group in range(first, stop):
                    kb = fx.Int32(group)
                    run_block(kb, left, right, kb % 64)
        else:
            for group in range(groups):
                run_block(fx.Int32(group), fx.Float32(0), fx.Float32(0), fx.Int32(0))

        final_scale = Vec.filled(4, Vec(pending_scale[None, 0].load())[0], fx.Float32)
        for rep_n in range_constexpr(2):
            index = last_pair[rep_n]
            c_acc[None, index].store(
                fx.fma(
                    Vec(pending[None, rep_n].load()),
                    final_scale,
                    Vec(c_acc[None, index].load()),
                )
            )

        if wave_m == 0:
            barrier()
        row_lane = wave_m * 16 + lane % 16
        if const_expr(lds_out):
            # The main loop has finished using A/B. Each 32 KiB region can
            # hold 64 output rows. Write two packets, then read them linearly
            # in 16-byte units before reusing the same LDS for the next pair.
            for half in range_constexpr(2):
                fx.gpu.barrier()
                for stage in range_constexpr(half * 2, half * 2 + 2):
                    buf = lds.a.ptr if const_expr(stage % 2 == 0) else lds.b.ptr
                    for rep_m in range_constexpr(reps[stage]):
                        row = row_lane + rep_m * 32
                        for side in range_constexpr(2):
                            for rep_n in range_constexpr(2):
                                index = ((stage * 2 + side) * 2 + rep_m) * 2 + rep_n
                                col = b_group(wave_n, side, rep_n) * 16 + lane // 16 * 4
                                word = col // 2
                                slot = ((word // 4) ^ (row & 31)) * 4 + word % 4
                                dtype = (
                                    fx.Float16
                                    if const_expr(fp16_output)
                                    else fx.BFloat16
                                )
                                packed = (
                                    Vec(c_acc[None, index].load())
                                    .to(dtype)
                                    .bitcast(fx.Int32)
                                )
                                fx.ptr_store(packed, buf + row * 128 + slot)
                fx.gpu.barrier()
                for stage in range_constexpr(half * 2, half * 2 + 2):
                    buf = lds.a.ptr if const_expr(stage % 2 == 0) else lds.b.ptr
                    for turn in range_constexpr(reps[stage] * 2):
                        word = (tid + turn * 512) * 4
                        row, col = word // 128, word % 128
                        slot = ((col // 4) ^ (row & 31)) * 4
                        value = fx.ptr_load(
                            buf + row * 128 + slot,
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                        offset = (starts[stage] + row) * N * 2 + col_base * 2 + col * 4
                        # Raw stores preserve descriptor tail masking and cache policy.
                        rocdl.raw_ptr_buffer_store(
                            value.ir_value(),
                            c_rsrc,
                            offset.ir_value(),
                            fx.Int32(0).ir_value(),
                            aux=ir.IntegerAttr.get(T.i32, cache_policy),
                        )
        elif const_expr(vector_store):
            # Bring the neighboring four-column fragment from the lane 16 positions
            # away. Each lane then writes eight adjacent BF16/FP16 values (16 B).
            odd = (lane // 16) % 2 == 1
            partner = (lane ^ 16) * 4
            for stage in range_constexpr(4):
                for rep_m in range_constexpr(reps[stage]):
                    row = row_lane + starts[stage] + rep_m * 32
                    for side in range_constexpr(2):
                        index = ((stage * 2 + side) * 2 + rep_m) * 2
                        dtype = fx.Float16 if const_expr(fp16_output) else fx.BFloat16
                        left = (
                            Vec(c_acc[None, index].load()).to(dtype).bitcast(fx.Int32)
                        )
                        right = (
                            Vec(c_acc[None, index + 1].load())
                            .to(dtype)
                            .bitcast(fx.Int32)
                        )
                        exchanged = [
                            fx.Int32(
                                rocdl.ds_bpermute(
                                    T.i32, partner, odd.select(left[j], right[j])
                                )
                            )
                            for j in range_constexpr(2)
                        ]
                        packed = Vec.from_elements(
                            [
                                odd.select(exchanged[j], left[j])
                                for j in range_constexpr(2)
                            ]
                            + [
                                odd.select(right[j], exchanged[j])
                                for j in range_constexpr(2)
                            ],
                            fx.Int32,
                        )
                        col = (
                            col_base
                            + b_group(wave_n, side, (lane // 16) % 2) * 16
                            + (lane // 32) * 8
                        )
                        rocdl.raw_ptr_buffer_store(
                            packed.ir_value(),
                            c_rsrc,
                            (row * (N * 2) + col * 2).ir_value(),
                            fx.Int32(0).ir_value(),
                            aux=ir.IntegerAttr.get(T.i32, vector_store_policy),
                        )
        else:
            col_lane = col_base + wave_n * 16 + lane // 16 * 4
            for index in range_constexpr(32):
                stage, side, rep_m, rep_n = (
                    index // 8,
                    index // 4 % 2,
                    index // 2 % 2,
                    index % 2,
                )
                row = row_lane + (stage * PACKET_ROWS + rep_m * 32)
                col = col_lane + (side * 128 + rep_n * 64)
                if const_expr(split_k):
                    # Two independent K halves write FP32 partials in tile-major order.
                    # Each writer owns its slot; final conversion follows the FP32 sum.
                    tile = block_m * n_tiles + block_n
                    local = row * TILE_N + col - col_base
                    fx.ptr_store(
                        Vec(c_acc[None, index].load()),
                        pC + fx.Int64(tile * 2 + part) * (TILE_M * TILE_N) + local,
                    )
                else:
                    offset = (col < N).select(
                        row * (N * 2) + col * 2, fx.Int32(0x7FFFFFFF)
                    )
                    packed = (
                        Vec(c_acc[None, index].load())
                        .to(fx.Float16 if const_expr(fp16_output) else fx.BFloat16)
                        .bitcast(fx.Int32)
                    )
                    rocdl.raw_ptr_buffer_store(
                        packed.ir_value(),
                        c_rsrc,
                        offset.ir_value(),
                        fx.Int32(0).ir_value(),
                        aux=ir.IntegerAttr.get(T.i32, 0),
                    )

    if split_k:

        @flyc.kernel(name=kernel_name + "_reduce", known_block_size=[256, 1, 1])
        def reduce(out: fx.Tensor, partials: fx.Tensor):
            tile = fx.Int32(fx.block_idx.y)
            local = (fx.Int32(fx.block_idx.x) * 256 + fx.Int32(fx.thread_idx.x)) * 8
            src = fx.get_iter(partials) + fx.Int64(tile) * 2 * (TILE_M * TILE_N) + local
            vec = fx.Vector.make_type(8, fx.Float32)
            left = Vec(fx.ptr_load(src, result_type=vec))
            right = Vec(fx.ptr_load(src + TILE_M * TILE_N, result_type=vec))
            row = tile // n_tiles * TILE_M + local // TILE_N
            col = tile % n_tiles * TILE_N + local % TILE_N
            fx.ptr_store(
                (left + right).to(out.element_type),
                fx.get_iter(out) + fx.Int64(row) * N + col,
            )

        @flyc.jit
        def launch_split(
            a: fx.Tensor,
            b: fx.Tensor,
            out: fx.Tensor,
            sa: fx.Tensor,
            sb: fx.Tensor,
            M: fx.Int32,
            partials: fx.Tensor,
            stream: fx.Stream,
        ):
            tiles = M // TILE_M * n_tiles
            gemm_kernel(
                fx.recast_iter(fx.Int32, fx.get_iter(a)),
                fx.recast_iter(fx.Int32, fx.get_iter(b)),
                fx.get_iter(partials),
                fx.get_iter(sa),
                fx.get_iter(sb),
                M,
            ).launch(grid=(tiles, 2, 1), block=(512, 1, 1), stream=stream)
            reduce(out, partials).launch(
                grid=(TILE_M * TILE_N // (256 * 8), tiles, 1),
                block=(256, 1, 1),
                stream=stream,
            )

        return launch_split

    @flyc.jit
    def launch(
        a: fx.Tensor,
        b: fx.Tensor,
        out: fx.Tensor,
        sa: fx.Tensor,
        sb: fx.Tensor,
        M: fx.Int32,
        stream: fx.Stream,
    ):
        gemm_kernel(
            fx.recast_iter(fx.Int32, fx.get_iter(a)),
            fx.recast_iter(fx.Int32, fx.get_iter(b)),
            fx.recast_iter(fx.Int32, fx.get_iter(out)),
            fx.get_iter(sa),
            fx.get_iter(sb),
            M,
        ).launch(
            grid=((M + tile_m - 1) // tile_m * n_tiles, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    return launch
