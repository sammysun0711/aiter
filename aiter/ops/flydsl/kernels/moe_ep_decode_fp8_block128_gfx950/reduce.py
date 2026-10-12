# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP32 sum of valid rank-local BF16 expert partials for the M32 path."""

import functools
import flydsl.compiler as fc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo, range_constexpr
from flydsl.expr.typing import Vector as Vec
from .common import _buffer, _i32


@functools.lru_cache(None)
def local_reduce_launcher(hidden, topk):
    @fc.kernel(name="flydsl_moe_ep_decode_fp8_b128_gfx950_reduce", known_block_size=[256, 1, 1])
    def kernel(
        P: fx.Tensor,
        IDS: fx.Tensor,
        MASK: fx.Tensor,
        WEIGHTS: fx.Tensor,
        COUNT: fx.Tensor,
        OUT: fx.Tensor,
    ):
        row = fx.Int32(fx.block_idx.x)
        if row < _i32(_buffer(COUNT), 0, uniform=True):
            partial, ids, mask, out = (
                _buffer(P),
                _buffer(IDS),
                _buffer(MASK),
                _buffer(OUT),
            )
            weights = _buffer(WEIGHTS)
            col = fx.Int32(fx.block_idx.y) * 1024 + fx.Int32(fx.thread_idx.x) * 4
            total = fx.full(4, 0.0, fx.Float32)
            for k in range_constexpr(topk):
                expert = _i32(ids, row * topk + k, uniform=True)
                weight = _i32(weights, row * topk + k, uniform=True).bitcast(fx.Float32)
                # Padded routes can repeat an expert with zero weight. Sorting
                # need not materialize every such slot; never read its scratch.
                if (_i32(mask, expert, uniform=True) != 0) & (weight != 0):
                    value = Vec(
                        bo.buffer_load(
                            partial,
                            (row * topk + k) * hidden + col,
                            vec_width=4,
                            dtype=fx.BFloat16,
                        )
                    )
                    total = total + Vec.from_elements(
                        [fx.Float32(value[i]) for i in range_constexpr(4)], fx.Float32
                    )
            converted = Vec.from_elements(
                [fx.BFloat16(total[i]) for i in range_constexpr(4)], fx.BFloat16
            )
            bo.buffer_store(converted, out, row * hidden + col)

    @fc.jit
    def launch(
        P: fx.Tensor,
        IDS: fx.Tensor,
        MASK: fx.Tensor,
        WEIGHTS: fx.Tensor,
        COUNT: fx.Tensor,
        OUT: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(P, IDS, MASK, WEIGHTS, COUNT, OUT).launch(
            grid=(rows, hidden // 1024, 1), block=(256, 1, 1), stream=stream
        )

    return launch
