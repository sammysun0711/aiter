# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block-FP8 down projection with contiguous four-column MFMA outputs."""
import functools
import flydsl.compiler as fc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from .common import _buffer, _i32, _fragment, _dot, _scale


@functools.lru_cache(None)
def down_launcher(hidden, intermediate, topk, workers, block_m=32, block_n=64):
    num_m, num_n = block_m // 16, block_n // 16
    atomic = block_m == 16
    suffix = "atomic" if atomic else "partial"

    @fc.kernel(
        name=f"flydsl_moe_ep_decode_fp8_b128_gfx950_s2_m{block_m}n{block_n}k128_transpose_{suffix}",
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
        WEIGHTS: fx.Tensor,
        OUT: fx.Tensor,
    ):
        lane = fx.Int32(fx.thread_idx.x)
        row_lane, k_lane = lane % 16, lane // 16
        ar, wr, sar, swr = _buffer(A), _buffer(W), _buffer(SA), _buffer(SW)
        ids, experts, weights, out = _buffer(IDS), _buffer(EXPERTS), _buffer(WEIGHTS), _buffer(OUT)
        padded = _i32(_buffer(VALID), 0, uniform=True)
        received = _i32(_buffer(COUNT), 0, uniform=True)
        ntiles = hidden // block_n
        for task in range(fx.Int32(fx.block_idx.x), (padded // block_m) * ntiles, workers):
            mtile, nbase = task // ntiles, (task % ntiles) * block_n
            expert = _i32(experts, mtile, uniform=True)
            rows = [mtile * block_m + mi * 16 + row_lane for mi in range_constexpr(num_m)]
            routes = [_i32(ids, row) for row in rows]
            tokens = [route & 0xFFFFFF for route in routes]

            def load(kb):
                activations, weight_tiles, scales = [], [], []
                sw = _scale(
                    swr, (expert * (hidden // 128) + nbase // 128) * (intermediate // 128) + kb
                )
                for mi in range_constexpr(num_m):
                    activations.append(
                        _fragment(
                            ar,
                            rows[mi] * intermediate + kb * 128 + k_lane * 16,
                            64,
                            tokens[mi] < received,
                        )
                    )
                    scales.append(
                        fx.Float32(
                            bo.buffer_load(
                                sar,
                                rows[mi] * (intermediate // 128) + kb,
                                vec_width=1,
                                dtype=fx.Float32,
                                mask=tokens[mi] < received,
                            )
                        )
                    )
                for ni in range_constexpr(num_n):
                    n = nbase + ni * 16 + row_lane
                    offset = expert * hidden * intermediate + (n // 16) * intermediate * 16
                    offset += (kb * 8 + k_lane) * 256 + (n % 16) * 16
                    weight_tiles.append(_fragment(wr, offset, 1024, non_temporal=True))
                return (
                    Vec.from_elements(
                        [v[i] for v in activations for i in range_constexpr(8)], fx.Int32
                    ),
                    Vec.from_elements(
                        [v[i] for v in weight_tiles for i in range_constexpr(8)], fx.Int32
                    ),
                    Vec.from_elements(scales, fx.Float32),
                    sw,
                )

            def step(acc, av, wv, sa, sw):
                values = []
                for mi in range_constexpr(num_m):
                    a = Vec.from_elements([av[mi * 8 + i] for i in range_constexpr(8)], fx.Int32)
                    for ni in range_constexpr(num_n):
                        w = Vec.from_elements(
                            [wv[ni * 8 + i] for i in range_constexpr(8)], fx.Int32
                        )
                        old = Vec.from_elements(
                            [acc[(mi * num_n + ni) * 4 + i] for i in range_constexpr(4)], fx.Float32
                        )
                        # Exchanging operands gives each lane four adjacent N
                        # outputs for one token, sharing one activation scale.
                        value = old + _dot(w, a) * sa[mi] * sw
                        values.extend([value[i] for i in range_constexpr(4)])
                return Vec.from_elements(values, fx.Float32)

            acc = fx.full(num_m * num_n * 4, 0.0, fx.Float32)
            av, wv, sa, sw = load(fx.Int32(0))
            for kb in range(intermediate // 128 - 1):
                an, wn, san, swn = load(kb + 1)
                acc = step(acc, av, wv, sa, sw)
                av, wv, sa, sw = an, wn, san, swn
            acc = step(acc, av, wv, sa, sw)
            for mi in range_constexpr(num_m):
                weight = fx.Float32(
                    bo.buffer_load(weights, rows[mi], vec_width=1, dtype=fx.Float32)
                )
                if tokens[mi] < received:
                    for ni in range_constexpr(num_n):
                        values = [
                            fx.BFloat16(acc[(mi * num_n + ni) * 4 + i] * weight)
                            for i in range_constexpr(4)
                        ]
                        col = nbase + ni * 16 + k_lane * 4
                        if const_expr(atomic):
                            # Sorting clears the output. Zero-weight padding
                            # must not atomically accumulate a possible NaN.
                            if weight != 0:
                                for pair in range_constexpr(2):
                                    packed = Vec.from_elements(
                                        values[pair * 2 : pair * 2 + 2], fx.BFloat16
                                    )
                                    offset = (tokens[mi] * hidden + col + pair * 2) * 2
                                    rocdl.raw_ptr_buffer_atomic_fadd(
                                        packed, out, offset, fx.Int32(0), fx.Int32(0)
                                    )
                        else:
                            slot = routes[mi] >> 24
                            offset = (tokens[mi] * topk + slot) * hidden + col
                            bo.buffer_store(Vec.from_elements(values, fx.BFloat16), out, offset)

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
        WEIGHTS: fx.Tensor,
        OUT: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(A, W, SA, SW, IDS, EXPERTS, VALID, COUNT, WEIGHTS, OUT).launch(
            grid=(workers, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch
