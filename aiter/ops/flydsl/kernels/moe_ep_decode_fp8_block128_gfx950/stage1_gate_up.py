# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""First-stage alternatives: fused gate/up or projection plus SiLU/quant.

The host selects the alternative; both produce the same FP8 Q layout.
"""

import functools
import flydsl.compiler as fc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from .common import _buffer, _i32, _fragment, _dot, _maximum, _scale

LOG2_E = 1.4426950408889634


@functools.lru_cache(None)
def gate_up_fused_launcher(hidden, intermediate, topk, block_m=32):
    """First GEMM: gate/up, SiLU and group128 FP8 quantization in one kernel.

    Each block computes one M16/M32 by N128 task, as two N64 halves.
    The launch grid uses the physical capacity; GPU counts bound live work."""
    reduction = hidden
    output_dim = intermediate
    weight_n = 2 * intermediate
    output_tile = 128
    num_m = block_m // 16
    num_n = 4

    @fc.kernel(
        name=f"flydsl_moe_ep_decode_fp8_b128_gfx950_s1_m{block_m}n{output_tile}k128",
        known_block_size=[64, 1, 1],
    )
    def kernel(
        A: fx.Tensor,
        W: fx.Tensor,
        SA: fx.Tensor,
        SW: fx.Tensor,
        IDS: fx.Tensor,
        EXPERTS: fx.Tensor,
        VALID: fx.Tensor,
        COUNT: fx.Tensor,
        AUX: fx.Tensor,
        OUT: fx.Tensor,
    ):
        lane = fx.Int32(fx.thread_idx.x)
        (row_lane, k_lane) = (lane % 16, lane // 16)
        (ar, wr, sar, swr) = (_buffer(A), _buffer(W), _buffer(SA), _buffer(SW))
        (ids, experts) = (_buffer(IDS), _buffer(EXPERTS))
        (aux, out) = (_buffer(AUX), _buffer(OUT))
        padded = _i32(_buffer(VALID), 0, uniform=True)
        received = _i32(_buffer(COUNT), 0, uniform=True)
        ntiles = output_dim // output_tile

        def compute_task(task):
            (mtile, nbase) = (task // ntiles, task % ntiles * output_tile)
            expert = _i32(experts, mtile, uniform=True)
            # Input fragments gather token rows; result fragments own different rows.
            source = [
                _i32(ids, mtile * block_m + mi * 16 + row_lane) & 0xFFFFFF
                for mi in range_constexpr(num_m)
            ]
            rows = [
                mtile * block_m + mi * 16 + k_lane * 4 + i
                for mi in range_constexpr(num_m)
                for i in range_constexpr(4)
            ]
            routes = [_i32(ids, r) for r in rows]
            tokens = [r & 0xFFFFFF for r in routes]

            def load(kb, n_offset):
                (aa, bb, uu, sa) = ([], [], [], [])
                sw = _scale(
                    swr,
                    (expert * (weight_n // 128) + (nbase + n_offset) // 128) * (reduction // 128)
                    + kb,
                )
                su = _scale(
                    swr,
                    (expert * (weight_n // 128) + (nbase + n_offset + intermediate) // 128)
                    * (reduction // 128)
                    + kb,
                )
                for mi in range_constexpr(num_m):
                    arow = source[mi]
                    aa.append(
                        _fragment(
                            ar, arow * reduction + kb * 128 + k_lane * 16, 64, source[mi] < received
                        )
                    )
                    for i in range_constexpr(4):
                        r = tokens[mi * 4 + i]
                        sa.append(
                            fx.Float32(
                                bo.buffer_load(
                                    sar,
                                    r * (reduction // 128) + kb,
                                    vec_width=1,
                                    dtype=fx.Float32,
                                    mask=tokens[mi * 4 + i] < received,
                                )
                            )
                        )
                for ni in range_constexpr(num_n):
                    n = nbase + n_offset + ni * 16 + row_lane
                    offset = (
                        expert * weight_n * reduction
                        + n // 16 * reduction * 16
                        + (kb * 8 + k_lane) * 256
                        + n % 16 * 16
                    )
                    bb.append(_fragment(wr, offset, 1024, non_temporal=True))
                    uu.append(
                        _fragment(wr, offset + intermediate * reduction, 1024, non_temporal=True)
                    )
                av = Vec.from_elements([a[i] for a in aa for i in range_constexpr(8)], fx.Int32)
                bv = Vec.from_elements([b[i] for b in bb for i in range_constexpr(8)], fx.Int32)
                uv = Vec.from_elements([u[i] for u in uu for i in range_constexpr(8)], fx.Int32)
                return (av, bv, uv, Vec.from_elements(sa, fx.Float32), sw, su)

            def step(g, u, av, bv, uv, sa, sw, su):
                (gv, uv_out) = ([], [])
                for mi in range_constexpr(num_m):
                    a = Vec.from_elements([av[mi * 8 + i] for i in range_constexpr(8)], fx.Int32)
                    scale = Vec.from_elements(
                        [sa[mi * 4 + i] for i in range_constexpr(4)], fx.Float32
                    )
                    for ni in range_constexpr(num_n):
                        off = (mi * num_n + ni) * 4
                        b = Vec.from_elements(
                            [bv[ni * 8 + i] for i in range_constexpr(8)], fx.Int32
                        )
                        old = Vec.from_elements(
                            [g[off + i] for i in range_constexpr(4)], fx.Float32
                        )
                        result = old + _dot(a, b) * scale * sw
                        gv.extend([result[i] for i in range_constexpr(4)])
                        b_up = Vec.from_elements(
                            [uv[ni * 8 + i] for i in range_constexpr(8)], fx.Int32
                        )
                        old_up = Vec.from_elements(
                            [u[off + i] for i in range_constexpr(4)], fx.Float32
                        )
                        result_up = old_up + _dot(a, b_up) * scale * su
                        uv_out.extend([result_up[i] for i in range_constexpr(4)])
                return (Vec.from_elements(gv, fx.Float32), Vec.from_elements(uv_out, fx.Float32))

            def gemm(n_offset):
                (g, u) = (
                    fx.full(num_m * num_n * 4, 0.0, fx.Float32),
                    fx.full(num_m * num_n * 4, 0.0, fx.Float32),
                )
                (av, bv, uv, sa, sw, su) = load(fx.Int32(0), n_offset)
                for kb in range(reduction // 128 - 1):
                    (an, bn, un, san, swn, sun) = load(kb + 1, n_offset)
                    (g, u) = step(g, u, av, bv, uv, sa, sw, su)
                    (av, bv, uv, sa, sw, su) = (an, bn, un, san, swn, sun)
                return step(g, u, av, bv, uv, sa, sw, su)

            (g0, u0) = gemm(0)
            z0 = g0 / (1.0 + fx.exp2(-g0 * LOG2_E)) * u0
            (g1, u1) = gemm(64)
            z1 = g1 / (1.0 + fx.exp2(-g1 * LOG2_E)) * u1
            for mi in range_constexpr(num_m):
                for i in range_constexpr(4):
                    oi = mi * 4 + i
                    (values, peak) = ([], fx.Float32(0.0))
                    for ni in range_constexpr(8):
                        index = (mi * 4 + ni % 4) * 4 + i
                        value = z0[index] if const_expr(ni < 4) else z1[index]
                        value = fx.Float32(
                            fx.arith.select(tokens[oi] < received, value, fx.Float32(0.0))
                        )
                        values.append(value)
                        peak = _maximum(peak, _maximum(value, -value))
                    # Sixteen lanes share one row's 128-channel quantization group.
                    for shift in range_constexpr(4):
                        peak = _maximum(peak, peak.shuffle_xor(fx.Int32(1 << shift), fx.Int32(16)))
                    scale = _maximum(peak, fx.Float32(1e-10)) / 448.0
                    if row_lane == 0:
                        bo.buffer_store(scale, aux, rows[oi] * (intermediate // 128) + nbase // 128)
                    for ni in range_constexpr(8):
                        q = fx.Int32(
                            rocdl.cvt_pk_fp8_f32(
                                fx.Int32.ir_type,
                                values[ni] / scale,
                                fx.Float32(0.0),
                                fx.Int32(0),
                                False,
                            )
                        )
                        bo.buffer_store(
                            fx.Int8(q), out, rows[oi] * intermediate + nbase + ni * 16 + row_lane
                        )

        task = fx.Int32(fx.block_idx.x)
        if task < padded // block_m * ntiles:
            compute_task(task)

    @fc.jit
    def launch(
        A: fx.Tensor,
        W: fx.Tensor,
        SA: fx.Tensor,
        SW: fx.Tensor,
        IDS: fx.Tensor,
        EXPERTS: fx.Tensor,
        VALID: fx.Tensor,
        COUNT: fx.Tensor,
        AUX: fx.Tensor,
        OUT: fx.Tensor,
        stream: fx.Stream,
    ):
        capacity = fx.Int32(fx.get_scalar(fx.size(A.shape))) // hidden
        local_experts = fx.Int32(fx.get_scalar(fx.size(W.shape))) // (2 * intermediate * hidden)
        max_tiles = (capacity * topk + local_experts * (block_m - 1) + block_m - 1) // block_m
        grid_x = max_tiles * (intermediate // 128)
        kernel(A, W, SA, SW, IDS, EXPERTS, VALID, COUNT, AUX, OUT).launch(
            grid=(grid_x, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


@functools.lru_cache(None)
def gate_up_projection_launcher(hidden, intermediate, m, capacity, splits, workers, n_tile):
    """Small-M first stage: write FP32 projections; the host uses M16/N64/splits=1."""
    num_m = m // 16
    num_n = n_tile // 16
    ntiles = 2 * intermediate // n_tile
    blocks_k = hidden // 128 // splits

    @fc.kernel(
        name=f"flydsl_moe_ep_decode_fp8_b128_gfx950_s1_m{m}n{n_tile}k128_proj_sk{splits}",
        known_block_size=[64, 1, 1],
    )
    def kernel(
        A: fx.Tensor,
        W: fx.Tensor,
        SA: fx.Tensor,
        SW: fx.Tensor,
        IDS: fx.Tensor,
        EXPERTS: fx.Tensor,
        VALID: fx.Tensor,
        COUNT: fx.Tensor,
        OUT: fx.Tensor,
    ):
        lane = fx.Int32(fx.thread_idx.x)
        row_lane = lane % 16
        k_lane = lane // 16
        ar, wr, sar, swr = _buffer(A), _buffer(W), _buffer(SA), _buffer(SW)
        ids, experts, out = _buffer(IDS), _buffer(EXPERTS), _buffer(OUT)
        count = _i32(_buffer(COUNT), 0, uniform=True)
        padded = _i32(_buffer(VALID), 0, uniform=True)
        for task in range(fx.Int32(fx.block_idx.x), (padded // m) * ntiles * splits, workers):
            split = task % splits
            tile = task // splits
            mtile = tile // ntiles
            nbase = (tile % ntiles) * n_tile
            expert = _i32(experts, mtile, uniform=True)
            source = [
                _i32(ids, mtile * m + mi * 16 + row_lane) & 0xFFFFFF
                for mi in range_constexpr(num_m)
            ]
            rows = [
                mtile * m + mi * 16 + k_lane * 4 + i
                for mi in range_constexpr(num_m)
                for i in range_constexpr(4)
            ]
            tokens = [_i32(ids, row) & 0xFFFFFF for row in rows]

            def load(kb):
                kg = split * blocks_k + kb
                aa = [
                    _fragment(ar, token * hidden + kg * 128 + k_lane * 16, 64, token < count)
                    for token in source
                ]
                ss = [
                    fx.Float32(
                        bo.buffer_load(
                            sar,
                            t * (hidden // 128) + kg,
                            vec_width=1,
                            dtype=fx.Float32,
                            mask=t < count,
                        )
                    )
                    for t in tokens
                ]
                sw = fx.Float32(
                    bo.buffer_load(
                        swr,
                        (expert * (2 * intermediate // 128) + nbase // 128) * (hidden // 128) + kg,
                        vec_width=1,
                        dtype=fx.Float32,
                    )
                )
                bb = []
                for ni in range_constexpr(num_n):
                    n = nbase + ni * 16 + row_lane
                    offset = (
                        expert * 2 * intermediate * hidden
                        + (n // 16) * hidden * 16
                        + (kg * 8 + k_lane) * 256
                        + (n % 16) * 16
                    )
                    bb.append(_fragment(wr, offset, 1024, non_temporal=True))
                return (
                    Vec.from_elements([v[i] for v in aa for i in range_constexpr(8)], fx.Int32),
                    Vec.from_elements([v[i] for v in bb for i in range_constexpr(8)], fx.Int32),
                    Vec.from_elements(ss, fx.Float32),
                    sw,
                )

            def step(acc, av, bv, sa, sw):
                values = []
                for mi in range_constexpr(num_m):
                    a = Vec.from_elements([av[mi * 8 + i] for i in range_constexpr(8)], fx.Int32)
                    scale = Vec.from_elements(
                        [sa[mi * 4 + i] for i in range_constexpr(4)], fx.Float32
                    )
                    for ni in range_constexpr(num_n):
                        b = Vec.from_elements(
                            [bv[ni * 8 + i] for i in range_constexpr(8)], fx.Int32
                        )
                        old = Vec.from_elements(
                            [acc[(mi * num_n + ni) * 4 + i] for i in range_constexpr(4)], fx.Float32
                        )
                        v = old + _dot(a, b) * scale * sw
                        values.extend([v[i] for i in range_constexpr(4)])
                return Vec.from_elements(values, fx.Float32)

            acc = fx.full(num_m * num_n * 4, 0.0, fx.Float32)
            av, bv, sa, sw = load(fx.Int32(0))
            for kb in range(blocks_k - 1):
                an, bn, san, swn = load(kb + 1)
                acc = step(acc, av, bv, sa, sw)
                av, bv, sa, sw = an, bn, san, swn
            acc = step(acc, av, bv, sa, sw)
            for mi in range_constexpr(num_m):
                for ni in range_constexpr(num_n):
                    for i in range_constexpr(4):
                        offset = (
                            (split * capacity + rows[mi * 4 + i]) * 2 * intermediate
                            + nbase
                            + ni * 16
                            + row_lane
                        )
                        bo.buffer_store(acc[(mi * num_n + ni) * 4 + i], out, offset)

    @fc.jit
    def launch(
        A: fx.Tensor,
        W: fx.Tensor,
        SA: fx.Tensor,
        SW: fx.Tensor,
        IDS: fx.Tensor,
        EXPERTS: fx.Tensor,
        VALID: fx.Tensor,
        COUNT: fx.Tensor,
        OUT: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(A, W, SA, SW, IDS, EXPERTS, VALID, COUNT, OUT).launch(
            grid=(workers, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch


@functools.lru_cache(None)
def silu_quant_launcher(intermediate, capacity, splits, workers):
    """Each wave computes SiLU and FP8 quantization for one row and128 channels."""
    groups = intermediate // 128

    @fc.kernel(
        name=f"flydsl_moe_ep_decode_fp8_b128_gfx950_silu_quant_sk{splits}",
        known_block_size=[64, 1, 1],
    )
    def kernel(
        P: fx.Tensor,
        IDS: fx.Tensor,
        VALID: fx.Tensor,
        COUNT: fx.Tensor,
        Q: fx.Tensor,
        SCALE: fx.Tensor,
    ):
        lane = fx.Int32(fx.thread_idx.x)
        pr, ids, qr, sr = _buffer(P), _buffer(IDS), _buffer(Q), _buffer(SCALE)
        padded = _i32(_buffer(VALID), 0, uniform=True)
        count = _i32(_buffer(COUNT), 0, uniform=True)
        for task in range(fx.Int32(fx.block_idx.x), padded * groups, workers):
            row = task // groups
            base = (task % groups) * 128
            token = _i32(ids, row, uniform=True) & 0xFFFFFF
            if token < count:
                values = []
                for side in range_constexpr(2):
                    gate = fx.Float32(0)
                    up = fx.Float32(0)
                    col = base + lane + side * 64
                    for split in range_constexpr(splits):
                        offset = (split * capacity + row) * 2 * intermediate + col
                        gate = gate + fx.Float32(
                            bo.buffer_load(pr, offset, vec_width=1, dtype=fx.Float32)
                        )
                        up = up + fx.Float32(
                            bo.buffer_load(pr, offset + intermediate, vec_width=1, dtype=fx.Float32)
                        )
                    values.append(gate / (1.0 + fx.exp2(-gate * 1.4426950408889634)) * up)
                peak = _maximum(_maximum(values[0], -values[0]), _maximum(values[1], -values[1]))
                for shift in range_constexpr(6):
                    peak = _maximum(peak, peak.shuffle_xor(fx.Int32(1 << shift), fx.Int32(64)))
                scale = _maximum(peak, fx.Float32(1e-10)) / 448.0
                if lane == 0:
                    bo.buffer_store(scale, sr, row * groups + base // 128)
                pair = fx.Int32(
                    rocdl.cvt_pk_fp8_f32(
                        fx.Int32.ir_type, values[0] / scale, values[1] / scale, fx.Int32(0), False
                    )
                )
                bo.buffer_store(fx.Int8(pair), qr, row * intermediate + base + lane)
                bo.buffer_store(fx.Int8(pair >> 8), qr, row * intermediate + base + lane + 64)

    @fc.jit
    def launch(
        P: fx.Tensor,
        IDS: fx.Tensor,
        VALID: fx.Tensor,
        COUNT: fx.Tensor,
        Q: fx.Tensor,
        SCALE: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(P, IDS, VALID, COUNT, Q, SCALE).launch(
            grid=(workers, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch
