# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
# Extracted from FionaZZ92/FlyDSL commit 9e24c459e8e7fa125ed12af631e1c3e121c0e079.

"""Convert MXFP4 to FP8 with a power-of-two scale per 128x128 weight tile."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.mxfp4_fp8_common import (
    _scaled_fp4_to_fp8x8,
    _scaled_fp8_tables,
)
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import global_typed_ptr
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw


@functools.cache
def compile_mxfp4_to_fp8_128block(*, experts, n, k, fnuz=True):
    """Reduce tile scales, then convert contiguous K=32 groups in a second pass."""
    if min(experts, n, k) <= 0 or k % 32:
        raise ValueError("experts, n and k must be positive, with K divisible by 32")
    threads = 64
    groups_per_channel = k // 32
    n_tiles, k_tiles = (n + 127) // 128, (k + 127) // 128
    headroom = 5 if fnuz else 6

    @flyc.kernel
    def reduce_scales(scales: fx.Int64, tile_scales: fx.Int64):
        tile = fx.Int64(gpu.block_id("x"))
        lane = fx.Int32(gpu.thread_id("x"))
        expert = tile // fx.Int64(n_tiles * k_tiles)
        row_base = (tile // fx.Int64(k_tiles) % fx.Int64(n_tiles)) * fx.Int64(128)
        group_base = (tile % fx.Int64(k_tiles)) * fx.Int64(4)
        sp = global_typed_ptr(scales, T.i8, align=1)
        max_code = fx.Int32(0)
        for chunk in range(8):
            local = fx.Int64(chunk * threads) + fx.Int64(lane)
            row = row_base + local // fx.Int64(4)
            col = group_base + local % fx.Int64(4)
            if (row < fx.Int64(n)) & (col < fx.Int64(groups_per_channel)):
                group = (expert * fx.Int64(n) + row) * fx.Int64(
                    groups_per_channel
                ) + col
                code = fx.Int32(sp[group]) & fx.Int32(255)
                finite_code = (code == fx.Int32(255)).select(fx.Int32(0), code)
                max_code = (finite_code > max_code).select(finite_code, max_code)
        for offset in (1, 2, 4, 8, 16, 32):
            other = fx.Int32(
                rocdl.ds_bpermute(
                    T.i32, _raw((lane ^ fx.Int32(offset)) * fx.Int32(4)), _raw(max_code)
                )
            )
            max_code = (other > max_code).select(other, max_code)
        exponent = max_code - fx.Int32(127 + headroom)
        exponent = (exponent < fx.Int32(-126)).select(fx.Int32(-126), exponent)
        if lane == fx.Int32(0):
            scale = ((exponent + fx.Int32(127)) << fx.Int32(23)).bitcast(fx.Float32)
            fx.ptr_store(
                scale, fx.add_offset(global_typed_ptr(tile_scales, T.f32), tile)
            )

    @flyc.kernel
    def convert(src: fx.Int64, scales: fx.Int64, dst: fx.Int64, tile_scales: fx.Int64):
        group = fx.Int64(gpu.block_id("x")) * fx.Int64(256) + fx.Int64(
            gpu.thread_id("x")
        )
        if group < fx.Int64(experts * n * groups_per_channel):
            channel = group // fx.Int64(groups_per_channel)
            tile = channel // fx.Int64(n) * fx.Int64(n_tiles) + channel % fx.Int64(
                n
            ) // fx.Int64(128)
            tile = tile * fx.Int64(k_tiles) + group % fx.Int64(
                groups_per_channel
            ) // fx.Int64(4)
            scale = fx.Float32(global_typed_ptr(tile_scales, T.f32)[tile])
            exponent = (scale.bitcast(fx.Int32) >> fx.Int32(23)) - fx.Int32(127)
            lo, hi, invalid = _scaled_fp8_tables(
                global_typed_ptr(scales, T.i8, align=1)[group],
                fnuz=fnuz,
                exponent_offset=exponent,
            )
            wp = global_typed_ptr(src, T.i32)
            dp = global_typed_ptr(dst, T.i32)
            packed = fx.Vector(
                fx.ptr_load(
                    fx.add_offset(wp, group * fx.Int64(4)), result_type=T.vec(4, T.i32)
                )
            )
            output = []
            for j in range_constexpr(4):
                converted = _scaled_fp4_to_fp8x8(
                    packed[j], lo, hi, invalid, fnuz=fnuz
                ).bitcast(fx.Int32)
                output.extend([_raw(converted[0]), _raw(converted[1])])
            fx.ptr_store(
                fx.Vector.from_elements(output, fx.Int32),
                fx.add_offset(dp, group * fx.Int64(8)),
            )

    @flyc.jit
    def launch(
        src: fx.Int64,
        scales: fx.Int64,
        dst: fx.Int64,
        tile_scales: fx.Int64,
        stream: fx.Stream,
    ):
        reduce_scales(scales, tile_scales).launch(
            grid=(experts * n_tiles * k_tiles, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )
        convert(src, scales, dst, tile_scales).launch(
            grid=((experts * n * groups_per_channel + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch
