# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Vectorized FP32 sum of valid rank-local BF16 expert partials."""

import functools
import flydsl.compiler as fc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo, const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec
from .common import _buffer, _i32
from flydsl._mlir.dialects import llvm


@functools.lru_cache(None)
def local_reduce_launcher(hidden, topk, route_masks=False):
    # Eight BF16 values form a 16-byte load/store. A CTA covers 2048 columns
    # when possible, halving the number of CTAs and repeated routing checks.
    width = 8
    threads = 256 if hidden % 2048 == 0 else 128
    vector_metadata = not route_masks and topk % 4 == 0

    @fc.kernel(
        name=f"flydsl_moe_ep_decode_fp8_b128_gfx950_reduce_vector8_rm{int(route_masks)}",
        known_block_size=[threads, 1, 1],
    )
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
            local_routes = _i32(ids, row, uniform=True) if route_masks else fx.Int32(0)
            col = (
                fx.Int32(fx.block_idx.y) * (threads * width)
                + fx.Int32(fx.thread_idx.x) * width
            )
            total = fx.full(width, 0.0, fx.Float32)
            # Fetch four adjacent routes per scalar instruction; retain the
            # original top-k addition order and skip zero-weight scratch slots.
            all_ids = (
                [
                    Vec(
                        bo.buffer_load(
                            ids,
                            row * topk + i * 4,
                            vec_width=4,
                            dtype=fx.Int32,
                            is_scalar=True,
                        )
                    )
                    for i in range_constexpr(topk // 4)
                ]
                if vector_metadata
                else None
            )
            all_weights = (
                [
                    Vec(
                        bo.buffer_load(
                            weights,
                            row * topk + i * 4,
                            vec_width=4,
                            dtype=fx.Int32,
                            is_scalar=True,
                        )
                    )
                    for i in range_constexpr(topk // 4)
                ]
                if vector_metadata
                else None
            )

            def has_partial(k):
                expert = (
                    fx.Int32(all_ids[k // 4][k % 4])
                    if vector_metadata
                    else _i32(ids, row * topk + k, uniform=True)
                )
                weight = (
                    fx.Int32(all_weights[k // 4][k % 4]).bitcast(fx.Float32)
                    if vector_metadata
                    else _i32(weights, row * topk + k, uniform=True).bitcast(fx.Float32)
                )
                return (_i32(mask, expert, uniform=True) != 0) & (weight != 0)

            def add_partial(k, total):
                value = Vec(
                    bo.buffer_load(
                        partial,
                        (row * topk + k) * hidden + col,
                        vec_width=width,
                        dtype=fx.BFloat16,
                    )
                )
                return total + Vec.from_elements(
                    [fx.Float32(value[i]) for i in range_constexpr(width)], fx.Float32
                )

            if const_expr(route_masks):
                remaining = local_routes
                while remaining != 0:
                    k = fx.Int32(llvm.intr_cttz(remaining.ir_value(), False))
                    total = add_partial(k, total)
                    remaining = remaining & (remaining - 1)
            else:
                for k in range_constexpr(topk):
                    if has_partial(k):
                        total = add_partial(k, total)
            converted = Vec.from_elements(
                [fx.BFloat16(total[i]) for i in range_constexpr(width)], fx.BFloat16
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
            grid=(rows, hidden // (threads * width), 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch
