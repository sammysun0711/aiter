# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Register fragments, buffer addressing and scaled FP8 MFMA primitives."""

import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm


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


def _fragment(
    resource,
    byte_offset,
    second_half,
    mask=None,
    non_temporal=False,
    soffset_bytes=None,
):
    cache = 2 if non_temporal else 0  # FlyDSL buffer-copy policy:2 = non-temporal.
    lo = Vec(
        bo.buffer_load(
            resource,
            byte_offset // 4,
            vec_width=4,
            dtype=fx.Int32,
            mask=mask,
            cache_modifier=cache,
            soffset_bytes=soffset_bytes,
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
            soffset_bytes=soffset_bytes,
        )
    )
    return Vec.from_elements(
        [lo[i] for i in range_constexpr(4)] + [hi[i] for i in range_constexpr(4)],
        fx.Int32,
    )


def _streaming_fragment(resource, byte_offset, second_half, soffset_bytes):
    """Non-temporal weight read for the adaptive cache branch.

    Use the descriptor-form ROCDL operation here and the pointer form for cached
    reads. Otherwise SCF can merge the two cache constants into a PHI for short
    K loops, violating the buffer intrinsic's immediate-argument requirement.
    Both forms lower to the same hardware buffer-load instruction family.
    """
    words = llvm.bitcast(
        Vec.make_type(4, fx.Int32),
        llvm.ptrtoint(ir.IntegerType.get_signless(128), resource),
    )
    words = words.ir_value() if hasattr(words, "ir_value") else words
    pieces = [
        Vec(
            rocdl.raw_buffer_load(
                Vec.make_type(4, fx.Int32),
                words,
                fx.Int32(byte_offset + delta).ir_value(),
                fx.Int32(soffset_bytes).ir_value(),
                fx.Int32(2).ir_value(),
            )
        )
        for delta in (0, second_half)
    ]
    return Vec.from_elements(
        [part[i] for part in pieces for i in range_constexpr(4)], fx.Int32
    )


def _weight_fragment(
    resource,
    byte_offset,
    second_half,
    *,
    scalar_base=None,
    cached=False,
    adaptive=False
):
    if adaptive and not cached:
        return _streaming_fragment(
            resource,
            byte_offset,
            second_half,
            0 if scalar_base is None else scalar_base,
        )
    return _fragment(
        resource,
        byte_offset,
        second_half,
        non_temporal=not cached,
        soffset_bytes=scalar_base,
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


# Routing ABI shared by local sorting and the GEMM/reduction consumers.
# OPUS emits plain expert IDs. Only local descriptors may carry EXPERT_REUSE.
TOKEN_MASK = 0xFFFFFF
SLOT_SHIFT = 24
EXPERT_MASK = 0xFFFF
EXPERT_REUSE = 1 << 16
PADDED_COUNT = 0
RECEIVED_COUNT = 1
ROUTE_MASK_OFFSET = 2


def _route_token(packed):
    return packed & TOKEN_MASK


def _route_slot(packed):
    return packed >> SLOT_SHIFT


def _pack_route(token, slot):
    return token | (slot << SLOT_SHIFT)


def _expert_id(descriptor, has_metadata):
    return descriptor & EXPERT_MASK if has_metadata else descriptor


def _expert_reuse(descriptor):
    return (descriptor & EXPERT_REUSE) != 0
