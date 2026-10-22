# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
# Extracted from FionaZZ92/FlyDSL commit 9e24c459e8e7fa125ed12af631e1c3e121c0e079.
"""Register-only E2M1/E8M0 to FP8 conversion primitives."""

import flydsl.expr as fx
from flydsl.expr import const_expr, rocdl

from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw


def _fp4_perm(src_hi, src_lo, sel):
    """v_perm_b32: result byte i = pool[sel.byte(i)], pool = {src_hi:src_lo}.

    Selector bytes 0..3 index src_lo bytes 0..3, selector bytes 4..7 index src_hi.
    """
    return fx.Int32(
        rocdl.perm_b32(_raw(fx.Int32(src_hi)), _raw(fx.Int32(src_lo)), _raw(sel))
    )


def _scaled_fp8_tables(scale_code, *, fnuz, exponent_offset=0):
    """Build a magnitude LUT for E2M1 * 2**(scale_code-127-exponent_offset).

    E8M0 scaling shifts the FP8 exponent exactly. Only subnormal outputs
    need rounding. Building the LUT once per 32-value block reuses this work for
    all 32 weights and avoids unsupported generic FP8 casts on gfx942.
    """
    code = fx.Int32(scale_code) & fx.Int32(255)
    # Keep the relative exponent signed: per-channel normalization can push
    # it below the E8M0 encoding range. Mask only the original input byte.
    delta = code - fx.Int32(127) - fx.Int32(exponent_offset)
    nan_code = 128 if fnuz else 127
    invalid = code == fx.Int32(255)
    lut = [invalid.select(fx.Int32(nan_code), fx.Int32(0))]
    for magnitude in (
        (56, 64, 68, 72, 76, 80, 84) if fnuz else (48, 56, 60, 64, 68, 72, 76)
    ):
        exponent = fx.Int32(magnitude >> 3) + delta
        normal = fx.Int32(magnitude) + delta * fx.Int32(8)
        # A subnormal is the four-bit significand shifted by 1-exponent.
        # Clamp before shifting: shifts >= 5 already round every value to 0.
        shift = fx.Int32(1) - exponent
        shift = (shift < fx.Int32(1)).select(fx.Int32(1), shift)
        shift = (shift > fx.Int32(5)).select(fx.Int32(5), shift)
        significand = fx.Int32(8 | (magnitude & 7))
        halfway = fx.Int32(1) << (shift - fx.Int32(1))
        tie_bit = significand.shrui(shift) & fx.Int32(1)
        subnormal = (significand + halfway - fx.Int32(1) + tie_bit).shrui(shift)
        value = (exponent <= fx.Int32(0)).select(subnormal, normal)
        overflow = (normal > fx.Int32(127 if fnuz else 126)) | invalid
        lut.append(overflow.select(fx.Int32(nan_code), value))
    packed = []
    for word in range(2):
        value = lut[word * 4]
        for j in range(1, 4):
            value = value | (lut[word * 4 + j] << fx.Int32(j * 8))
        packed.append(value)
    return packed[0], packed[1], invalid


def _scaled_fp4_to_fp8x8(raw_i32, lo, hi, invalid_scale, *, fnuz):
    raw = fx.Int32(raw_i32)
    even = _fp4_perm(hi, lo, raw & fx.Int32(0x07070707))
    odd = _fp4_perm(hi, lo, raw.shrui(fx.Int32(4)) & fx.Int32(0x07070707))
    sign_even = (raw & fx.Int32(0x08080808)) << fx.Int32(4)
    sign_odd = raw & fx.Int32(0x80808080)
    if const_expr(fnuz):
        # Bit 7 survives iff the magnitude is nonzero. No cross-byte borrow.
        # Underflowed negative values become +0; a LUT NaN remains 0x80.
        nz_even = (even | fx.Int32(0x80808080)) - fx.Int32(0x01010101)
        nz_odd = (odd | fx.Int32(0x80808080)) - fx.Int32(0x01010101)
        sign_even = sign_even & nz_even
        sign_odd = sign_odd & nz_odd
    else:
        # E8M0 NaN has no sign. Finite scales preserve FN signed zero/NaN.
        sign_even = invalid_scale.select(fx.Int32(0), sign_even)
        sign_odd = invalid_scale.select(fx.Int32(0), sign_odd)
    even = even | sign_even
    odd = odd | sign_odd
    p0 = _fp4_perm(odd, even, fx.Int32(0x05010400))
    p1 = _fp4_perm(odd, even, fx.Int32(0x07030602))
    dtype = fx.Float8E4M3FNUZ if fnuz else fx.Float8E4M3FN
    return fx.Vector.from_elements([_raw(p0), _raw(p1)], fx.Int32).bitcast(dtype)
