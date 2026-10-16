# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Register fragments, buffer addressing and scaled FP8 MFMA primitives."""

import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec


def _buffer(tensor):
    # max_size=False alone falls back to4GiB for fly.memref. Exact lengths
    # are essential: masked raw loads use an out-of-range sentinel offset.
    elements = fx.Int64(fx.get_scalar(fx.size(tensor.shape)))
    return bo.create_buffer_resource(
        tensor, num_records_bytes=elements * (tensor.element_type.ir_type.width // 8)
    )


def _i32(resource, offset, uniform=False):
    return fx.Int32(
        bo.buffer_load(resource, offset, vec_width=1, dtype=fx.Int32, is_scalar=uniform)
    )


def _scale(resource, offset):
    # Weight-scale addresses are wave-uniform; activation-scale addresses are not.
    return _i32(resource, offset, uniform=True).bitcast(fx.Float32)


def _fragment(resource, byte_offset, second_half, mask=None, non_temporal=False):
    cache = 2 if non_temporal else 0  # FlyDSL buffer-copy policy:2 = non-temporal.
    lo = Vec(
        bo.buffer_load(
            resource,
            byte_offset // 4,
            vec_width=4,
            dtype=fx.Int32,
            mask=mask,
            cache_modifier=cache,
        )
    )
    hi = Vec(
        bo.buffer_load(
            resource,
            (byte_offset + second_half) // 4,
            vec_width=4,
            dtype=fx.Int32,
            mask=mask,
            cache_modifier=cache,
        )
    )
    return Vec.from_elements(
        [lo[i] for i in range_constexpr(4)] + [hi[i] for i in range_constexpr(4)],
        fx.Int32,
    )


def _dot(a, b):
    # Native FP8 MFMA; E8M0 scale bytes0x7f encode1. FP32 block scales
    # are applied separately to each128-wide K partial below.
    return Vec(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            Vec.make_type(4, fx.Float32),
            [a, b, fx.full(4, 0.0, fx.Float32), 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
        )
    )


def _maximum(a, b):
    return fx.Float32(fx.arith.maximumf(a.ir_value(), b.ir_value()))
