# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gate/Up GEMM and SiLU/FP8 quantization in separate GPU kernels.

The GEMM writes FP32 Gate/Up to a workspace. A second kernel reads it,
applies SiLU(gate) * up, and writes group128 FP8 values/scales for Down.
Both launchers live here because together they implement stage 1.
"""

import functools
import flydsl.compiler as fc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from .common import _buffer, _i32, _fragment, _weight_fragment, _dot, _maximum, _scale
from .configuration import WeightCache
from .common import PADDED_COUNT
from .common import _route_token, _expert_id


@functools.lru_cache(None)
def gate_up_gemm_launcher(
    hidden,
    intermediate,
    m,
    capacity,
    splits,
    workers,
    n_tile,
    cache_policy=WeightCache.STREAM,
    transpose=False,
    reuse_metadata=False,
):
    """Write FP32 Gate/Up using the physical sort tile and live M16 fragments.

    Counts remain on the GPU. Empty M16 fragments are skipped; padding rows
    within a live fragment are not stored. The quantizer reads valid rows only.
    """
    num_m = m // 16
    num_n = n_tile // 16
    ntiles = 2 * intermediate // n_tile
    blocks_k = hidden // 128 // splits

    @fc.kernel(
        name=f"flydsl_moe_ep_decode_fp8_b128_gfx950_s1_m{m}n{n_tile}k128_proj_sk{splits}_c{int(cache_policy)}_t{int(transpose)}_sa{int(transpose)}",
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
        padded = _i32(_buffer(VALID), PADDED_COUNT, uniform=True)
        for task in range(
            fx.Int32(fx.block_idx.x), (padded // m) * ntiles * splits, workers
        ):
            split = task % splits
            tile = task // splits
            mtile = tile // ntiles
            nbase = (tile % ntiles) * n_tile
            descriptor = _i32(experts, mtile, uniform=True)
            expert = _expert_id(descriptor, reuse_metadata)

            def compute(active_m):
                cached = (
                    (active_m == m // 16 and m >= 32)
                    if cache_policy == WeightCache.ADAPTIVE
                    else cache_policy
                )
                source = [
                    _route_token(_i32(ids, mtile * m + mi * 16 + row_lane))
                    for mi in range_constexpr(active_m)
                ]
                rows = (
                    [mtile * m + mi * 16 + row_lane for mi in range_constexpr(active_m)]
                    if transpose
                    else [
                        mtile * m + mi * 16 + k_lane * 4 + i
                        for mi in range_constexpr(active_m)
                        for i in range_constexpr(4)
                    ]
                )
                tokens = [_route_token(_i32(ids, row)) for row in rows]

                output_tokens = [_route_token(_i32(ids, row)) for row in rows]

                def load(kb):
                    kg = split * blocks_k + kb
                    aa = [
                        _fragment(
                            ar,
                            (
                                token * hidden + k_lane * 16
                                if transpose
                                else token * hidden + kg * 128 + k_lane * 16
                            ),
                            64,
                            token < count,
                            soffset_bytes=kg * 128 if transpose else None,
                        )
                        for token in source
                    ]
                    ss = [
                        fx.Float32(
                            bo.buffer_load(
                                sar,
                                (
                                    (t * (hidden // 128))
                                    if transpose
                                    else (t * (hidden // 128) + kg)
                                ),
                                soffset_bytes=(kg * (1) * 4 if transpose else None),
                                vec_width=1,
                                dtype=fx.Float32,
                                mask=t < count,
                            )
                        )
                        for t in tokens
                    ]
                    scale_offset = (
                        expert * (2 * intermediate // 128) + nbase // 128
                    ) * (hidden // 128) + kg
                    sw = (
                        _scale(swr, scale_offset)
                        if transpose
                        else fx.Float32(
                            bo.buffer_load(
                                swr, scale_offset, vec_width=1, dtype=fx.Float32
                            )
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
                        # Scalar tile/K bases leave only the packed lane offset in
                        # VGPRs. Keep the original addressing for untuned profiles.
                        scalar_base = expert * 2 * intermediate * hidden
                        scalar_base += (nbase // 16 + ni) * hidden * 16 + kg * 2048
                        bb.append(
                            _weight_fragment(
                                wr,
                                (row_lane * 16 + k_lane * 256 if transpose else offset),
                                1024,
                                cached=cached,
                                adaptive=cache_policy == WeightCache.ADAPTIVE,
                                scalar_base=scalar_base if transpose else None,
                            )
                        )
                    return (
                        Vec.from_elements(
                            [v[i] for v in aa for i in range_constexpr(8)], fx.Int32
                        ),
                        Vec.from_elements(
                            [v[i] for v in bb for i in range_constexpr(8)], fx.Int32
                        ),
                        Vec.from_elements(ss, fx.Float32),
                        sw,
                    )

                def step(acc, av, bv, sa, sw):
                    values = []
                    for mi in range_constexpr(active_m):
                        a = Vec.from_elements(
                            [av[mi * 8 + i] for i in range_constexpr(8)], fx.Int32
                        )
                        scale = (
                            sa[mi]
                            if transpose
                            else Vec.from_elements(
                                [sa[mi * 4 + i] for i in range_constexpr(4)], fx.Float32
                            )
                        )
                        for ni in range_constexpr(num_n):
                            b = Vec.from_elements(
                                [bv[ni * 8 + i] for i in range_constexpr(8)], fx.Int32
                            )
                            old = Vec.from_elements(
                                [
                                    acc[(mi * num_n + ni) * 4 + i]
                                    for i in range_constexpr(4)
                                ],
                                fx.Float32,
                            )
                            dot = _dot(b, a) if transpose else _dot(a, b)
                            v = old + dot * (scale * sw)
                            values.extend([v[i] for i in range_constexpr(4)])
                    return Vec.from_elements(values, fx.Float32)

                acc = fx.full(active_m * num_n * 4, 0.0, fx.Float32)
                av, bv, sa, sw = load(fx.Int32(0))
                for kb in range(blocks_k - 1):
                    an, bn, san, swn = load(kb + 1)
                    acc = step(acc, av, bv, sa, sw)
                    av, bv, sa, sw = an, bn, san, swn
                acc = step(acc, av, bv, sa, sw)
                for mi in range_constexpr(active_m):
                    for ni in range_constexpr(num_n):
                        if transpose:
                            offset = (
                                (split * capacity + rows[mi]) * 2 * intermediate
                                + nbase
                                + ni * 16
                                + k_lane * 4
                            )
                            value = Vec.from_elements(
                                [
                                    acc[(mi * num_n + ni) * 4 + i]
                                    for i in range_constexpr(4)
                                ],
                                fx.Float32,
                            )
                            bo.buffer_store(
                                value, out, offset, mask=output_tokens[mi] < count
                            )
                        else:
                            for i in range_constexpr(4):
                                offset = (
                                    (split * capacity + rows[mi * 4 + i])
                                    * 2
                                    * intermediate
                                    + nbase
                                    + ni * 16
                                    + row_lane
                                )
                                bo.buffer_store(
                                    acc[(mi * num_n + ni) * 4 + i],
                                    out,
                                    offset,
                                    mask=output_tokens[mi * 4 + i] < count,
                                )

            def has_rows(offset):
                return _route_token(_i32(ids, mtile * m + offset, uniform=True)) < count

            if const_expr(m == 16):
                compute(1)
            elif const_expr(m == 32):
                if has_rows(16):
                    compute(2)
                else:
                    compute(1)
            else:
                # Sparse experts need one count probe, rather than three
                # serialized scalar loads before entering their K loop.
                if has_rows(16):
                    if has_rows(32):
                        if has_rows(48):
                            compute(4)
                        else:
                            compute(3)
                    else:
                        compute(2)
                else:
                    compute(1)

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
def silu_quant_launcher(
    intermediate, capacity, splits, workers, block_m=32, tiled=False
):
    """Build linear or tile-grouped FP32 SiLU / group128 FP8 quantization."""
    if not tiled:
        return _silu_quant_linear_launcher(intermediate, capacity, splits, workers)
    groups = intermediate // 128
    workers //= 4

    @fc.kernel(
        name=f"flydsl_moe_ep_decode_fp8_b128_gfx950_silu_quant_sk{splits}",
        known_block_size=[256, 1, 1],
    )
    def kernel(
        P: fx.Tensor,
        IDS: fx.Tensor,
        VALID: fx.Tensor,
        COUNT: fx.Tensor,
        Q: fx.Tensor,
        SCALE: fx.Tensor,
    ):
        thread = fx.Int32(fx.thread_idx.x)
        lane = thread % 64
        wave = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, thread // 64))
        pr, ids, qr, sr = _buffer(P), _buffer(IDS), _buffer(Q), _buffer(SCALE)
        padded = _i32(_buffer(VALID), PADDED_COUNT, uniform=True)
        count = _i32(_buffer(COUNT), 0, uniform=True)
        for task in range(
            fx.Int32(fx.block_idx.x), (padded // block_m) * groups, workers
        ):
            tile_start = task // groups * block_m
            row = tile_start + wave
            base = (task % groups) * 128
            token = _route_token(_i32(ids, row, uniform=True))
            # Valid sorted rows precede tile padding. Four independent waves
            # advance through a tile and stop as soon as they reach its tail.
            while (row < tile_start + block_m) & (token < count):
                # Each lane loads two adjacent channels in one vector and
                # writes one packed FP8 pair. All 128 channels still share
                # the same maximum and scale, reduced across the same wave.
                gate = fx.full(2, 0.0, fx.Float32)
                up = fx.full(2, 0.0, fx.Float32)
                col = base + lane * 2
                for split in range_constexpr(splits):
                    offset = (split * capacity + row) * 2 * intermediate + col
                    gate = gate + Vec(
                        bo.buffer_load(pr, offset, vec_width=2, dtype=fx.Float32)
                    )
                    up = up + Vec(
                        bo.buffer_load(
                            pr, offset + intermediate, vec_width=2, dtype=fx.Float32
                        )
                    )
                values = gate / (1.0 + fx.exp2(-gate * 1.4426950408889634)) * up
                peak = _maximum(
                    _maximum(values[0], -values[0]), _maximum(values[1], -values[1])
                )
                for shift in range_constexpr(6):
                    peak = _maximum(
                        peak, peak.shuffle_xor(fx.Int32(1 << shift), fx.Int32(64))
                    )
                scale = _maximum(peak, fx.Float32(1e-10)) / 448.0
                if lane == 0:
                    bo.buffer_store(scale, sr, row * groups + base // 128)
                pair = fx.Int32(
                    rocdl.cvt_pk_fp8_f32(
                        fx.Int32.ir_type,
                        values[0] / scale,
                        values[1] / scale,
                        fx.Int32(0),
                        False,
                    )
                )
                bo.buffer_store(
                    fx.Int16(pair), qr, (row * intermediate + base) // 2 + lane
                )
                row = row + 4
                token = _route_token(_i32(ids, row, uniform=True))

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
            grid=(workers, 1, 1), block=(256, 1, 1), stream=stream
        )

    return launch


@functools.lru_cache(None)
def _silu_quant_linear_launcher(intermediate, capacity, splits, workers):
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
        padded = _i32(_buffer(VALID), PADDED_COUNT, uniform=True)
        count = _i32(_buffer(COUNT), 0, uniform=True)
        for task in range(fx.Int32(fx.block_idx.x), padded * groups, workers):
            row = task // groups
            base = (task % groups) * 128
            token = _route_token(_i32(ids, row, uniform=True))
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
                            bo.buffer_load(
                                pr, offset + intermediate, vec_width=1, dtype=fx.Float32
                            )
                        )
                    values.append(
                        gate / (1.0 + fx.exp2(-gate * 1.4426950408889634)) * up
                    )
                peak = _maximum(
                    _maximum(values[0], -values[0]), _maximum(values[1], -values[1])
                )
                for shift in range_constexpr(6):
                    peak = _maximum(
                        peak, peak.shuffle_xor(fx.Int32(1 << shift), fx.Int32(64))
                    )
                scale = _maximum(peak, fx.Float32(1e-10)) / 448.0
                if lane == 0:
                    bo.buffer_store(scale, sr, row * groups + base // 128)
                pair = fx.Int32(
                    rocdl.cvt_pk_fp8_f32(
                        fx.Int32.ir_type,
                        values[0] / scale,
                        values[1] / scale,
                        fx.Int32(0),
                        False,
                    )
                )
                bo.buffer_store(fx.Int8(pair), qr, row * intermediate + base + lane)
                bo.buffer_store(
                    fx.Int8(pair >> 8), qr, row * intermediate + base + lane + 64
                )

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
