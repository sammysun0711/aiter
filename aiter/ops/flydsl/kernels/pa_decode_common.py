# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Math and copy helpers for the PA decode kernels."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, rocdl
from flydsl.expr.typing import T

from .dpp_utils import update_dpp_i32


def global_pointer_from_addr(addr, dtype, *, alignment: int):
    ptr_type = fx.PointerType.get(
        elem_ty=dtype.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=alignment,
    )
    return fx.inttoptr(ptr_type, addr)


def copy_load(source, offset, copy_atom, register):
    fx.copy(copy_atom, fx.slice(source, (None, fx.Int32(offset))), register)
    return fx.memref_load_vec(register)


def copy_store(destination, offset, copy_atom, register, value):
    fx.memref_store_vec(value, register)
    fx.copy(copy_atom, register, fx.slice(destination, (None, fx.Int32(offset))))


def rcp_f32(value):
    return rocdl.rcp(T.f32, value)


def exp2_amdgcn_scalar(scalar_value):
    raw = fx.as_ir_value(scalar_value)
    f32_ty = ir.F32Type.get()
    return llvm.call_intrinsic(f32_ty, "llvm.amdgcn.exp2.f32", [raw], [], [])


def exp2_f32_fast(value):
    raw = fx.as_ir_value(value)
    ty = raw.type
    if isinstance(ty, ir.VectorType):
        vec = fx.Vector(raw)
        elems = [exp2_amdgcn_scalar(vec[i]) for i in range(ty.shape[0])]
        return fx.Vector.from_elements(elems, vec.dtype)
    return exp2_amdgcn_scalar(raw)


def cdiv(numer, denom):
    """Ceiling division for host integers and typed DSL integer values."""
    if isinstance(numer, (fx.Numeric, fx.Vector)) or isinstance(
        denom, (fx.Numeric, fx.Vector)
    ):
        return fx.ceildiv(numer, denom)
    return -(-numer // denom)


def pow2_shift(value: int) -> int:
    assert value > 0 and (value & (value - 1)) == 0
    return value.bit_length() - 1


def is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def udiv_pow2(value, divisor: int):
    return value >> fx.Int32(pow2_shift(divisor))


def urem_pow2(value, divisor: int):
    return value & fx.Int32(divisor - 1)


def udiv_const(value, divisor: int):
    if const_expr(is_pow2(divisor)):
        return udiv_pow2(value, divisor)
    return value // fx.Int32(divisor)


def urem_const(value, divisor: int):
    if const_expr(is_pow2(divisor)):
        return urem_pow2(value, divisor)
    return value % fx.Int32(divisor)


def dpp_xor_f32(src, offset: int, **kw):
    """Return ``src`` from the lane selected by a 16-lane XOR DPP pattern."""
    src_i32 = fx.Float32(src).bitcast(fx.Int32)
    if offset == 8:
        out_i32 = update_dpp_i32(src_i32, src_i32, 280, 0xF, 0xC, False, **kw)
        out_i32 = update_dpp_i32(out_i32, src_i32, 264, 0xF, 0x3, False, **kw)
    elif offset == 4:
        out_i32 = update_dpp_i32(src_i32, src_i32, 276, 0xF, 0xA, False, **kw)
        out_i32 = update_dpp_i32(out_i32, src_i32, 260, 0xF, 0x5, False, **kw)
    elif offset == 2:
        out_i32 = update_dpp_i32(src_i32, src_i32, 78, 0xF, 0xF, False, **kw)
    elif offset == 1:
        out_i32 = update_dpp_i32(src_i32, src_i32, 177, 0xF, 0xF, False, **kw)
    else:
        raise ValueError(
            f"dpp_xor_f32 only supports 16-lane offsets 1, 2, 4, 8; got {offset}"
        )
    return fx.Int32(out_i32).bitcast(fx.Float32).ir_value()
