# SPDX-License-Identifier: Apache-2.0
# Operand layout and output epilogues follow the pinned FlyDSL A16W4 kernels
# (Copyright (C) 2025-2026 FlyDSL Project Contributors).
"""gfx942 dynamic MXFP4 MoE using FP8 MFMA, without expanded model weights.

Weights stay E2M1/E8M0. Each call scans scale exponents (not weight values),
normalizes each channel by 2**(max_exponent-127-5), and decodes packed nibbles
directly to FP8 register operands. FP8 activations have per-row FP32 scales.
All scales/activation quantization/sorting/reduction are inside inference.
This is lossy A8W4, not bit-equivalent to the BF16-activation A16W4 operator.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
import torch

from ..tensor_shim import _run_compiled
from .utils import _channel_factor, _fp4x8_to_fp8x8, _fp8_tables, _mma_fp8

def _pointer(address, dtype, alignment=16):
    return fx.inttoptr(
        fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, alignment),
        fx.Int64(address),
    )

def _scale_index(row, group, groups):
    # shuffle_scale: [rows/32, groups/8, 4, 16, 2, 2].
    return (
        (row // 32) * (groups * 32)
        + (group // 8) * 256
        + (group % 4) * 64
        + (row % 16) * 4
        + ((group // 4) % 2) * 2
        + (row % 32) // 16
    )


def _pack_activation(a, b, c, d):
    values = [fx.max(fx.min(v, 240.0), -240.0) for v in (a, b, c, d)]
    lo = rocdl.cvt_pk_fp8_f32(T.i32, values[0], values[1], fx.Int32(0), False)
    packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, values[2], values[3], lo, True))
    mag = packed & fx.Uint32(0x7F7F7F7F)
    zero = (~(mag + fx.Uint32(0x7F7F7F7F))) & fx.Uint32(0x80808080)
    return (packed & ~zero).bitcast(fx.Int32)

def _magnitude(mag):
    bits = (mag + 252) << 22
    bits = (mag == 1).select(0x3F000000, bits)
    bits = (mag == 0).select(0, bits)
    return bits.bitcast(fx.Float32)

def _pack_fp8(word, shift, factor):
    values = []
    for j in range_constexpr(4):
        nibble = (word >> (shift + 4 * j)) & 15
        sign = ((nibble & 8) != 0).select(-factor, factor)
        value = _magnitude(nibble & 7) * sign
        values.append(fx.max(fx.min(value, 240.0), -240.0))
    low = rocdl.cvt_pk_fp8_f32(T.i32, values[0], values[1], fx.Int32(0), False)
    packed = fx.Uint32(rocdl.cvt_pk_fp8_f32(T.i32, values[2], values[3], low, True))
    # FNUZ has no negative zero. For finite, clamped input an encoded 0x80
    # can only be a rounded negative zero; clear its sign bit byte-wise.
    magnitude = packed & fx.Uint32(0x7F7F7F7F)
    zero_signs = (~(magnitude + fx.Uint32(0x7F7F7F7F))) & fx.Uint32(0x80808080)
    return (packed & ~zero_signs).bitcast(fx.Int32)

@cache
def _compile_row_quant(cols, rows, dynamic_rows):
    if cols % 256:
        raise ValueError("activation K must be a multiple of 256")
    chunks = cols // 256

    @flyc.kernel(known_block_size=(256, 1, 1))
    def kernel(source: fx.Int64, output: fx.Int64, scale: fx.Int64, valid: fx.Int64):
        tid = fx.Int32(gpu.thread_id("x"))
        row = fx.Int32(gpu.block_id("x")) * 4 + tid // 64
        lane = tid % 64
        count = fx.Int32(rows)
        if const_expr(dynamic_rows):
            count = fx.ptr_load(_pointer(valid, fx.Int32, 4))
        if row < count:
            values = []
            amax = fx.Float32(0.0)
            for j in range_constexpr(chunks):
                offset = fx.Int64(row) * cols + lane * 4 + j * 256
                v = fx.ptr_load(fx.add_offset(_pointer(source, fx.BFloat16, 8), offset),
                                result_type=fx.Vector.make_type(4, fx.BFloat16)).to(fx.Float32)
                values.append(v)
                for k in range_constexpr(4):
                    amax = fx.max(amax, fx.max(v[k], -v[k]))
            for bit in range_constexpr(6):
                amax = fx.max(amax, amax.shuffle_xor(1 << bit, 64))
            s = (amax == 0.0).select(fx.Float32(1.0), amax / 240.0)
            inv = 1.0 / s
            if lane == 0:
                fx.ptr_store(s, fx.add_offset(_pointer(scale, fx.Float32, 4), row))
            for j in range_constexpr(chunks):
                v = values[j] * inv
                packed = _pack_activation(v[0], v[1], v[2], v[3])
                offset = fx.Int64(row) * (cols // 4) + lane + j * 64
                fx.ptr_store(packed, fx.add_offset(_pointer(output, fx.Int32, 4), offset))

    @flyc.jit
    def launch(source: fx.Int64, output: fx.Int64, scale: fx.Int64, valid: fx.Int64, stream: fx.Stream):
        kernel(source, output, scale, valid).launch(grid=((rows + 3) // 4,), block=(256,), stream=stream)

    return launch


def quantize_rows(x, valid=None):
    rows, cols = x.shape
    q = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    scales = torch.empty(rows, device=x.device, dtype=torch.float32)
    _run_compiled(_compile_row_quant(cols, rows, valid is not None), x.data_ptr(), q.data_ptr(), scales.data_ptr(),
         valid.data_ptr() if valid is not None else 0, torch.cuda.current_stream())
    return q, scales


@cache
def _compile_channel_exponents(rows, cols):
    groups = cols // 32
    rows_per_wave = 4 if cols >= 4096 else 16
    partitions = 64 // rows_per_wave
    if groups % partitions:
        raise ValueError("unsupported scale group geometry")
    rounds = groups // partitions
    rows_per_block = 4 * rows_per_wave

    @flyc.kernel(known_block_size=(256, 1, 1))
    def kernel(scales: fx.Int64, output: fx.Int64):
        tid = fx.Int32(gpu.thread_id("x"))
        row = fx.Int32(gpu.block_id("x")) * rows_per_block + (tid // 64) * rows_per_wave + tid % rows_per_wave
        part = (tid % 64) // rows_per_wave
        if row < rows:
            largest = fx.Int32(0)
            for j in range(rounds):
                group = part + j * partitions
                code = fx.ptr_load(fx.add_offset(_pointer(scales, fx.Uint8, 1),
                                                 fx.Int64(_scale_index(row, group, groups))))
                largest = fx.max(largest, code.to(fx.Int32) & 255)
            for bit in range_constexpr(partitions.bit_length() - 1):
                largest = fx.max(largest, largest.shuffle_xor(rows_per_wave << bit, 64))
            if part == 0:
                fx.ptr_store(largest, fx.add_offset(_pointer(output, fx.Int32, 4), row))

    @flyc.jit
    def launch(scales: fx.Int64, output: fx.Int64, stream: fx.Stream):
        kernel(scales, output).launch(grid=((rows + rows_per_block - 1) // rows_per_block,), block=(256,), stream=stream)

    return launch


def channel_exponents(packed, scales):
    experts, n, kp = packed.shape
    result = torch.empty((experts, n), device=packed.device, dtype=torch.int32)
    _run_compiled(_compile_channel_exponents(experts * n, kp * 2), scales.data_ptr(), result.data_ptr(),
         torch.cuda.current_stream())
    return result


def _decode_fp8(raw, group_scale, max_code, fast_decode, tables=None):
    """Eight consecutive FP4 nibbles -> eight FP8 register values."""
    if const_expr(fast_decode):
        return _fp4x8_to_fp8x8(raw, tables if tables is not None else _fp8_tables(group_scale, max_code))
    raw = fx.Uint32(raw)
    code = (fx.Float32(group_scale).bitcast(fx.Uint32) >> 23).to(fx.Int32)
    difference = max_code - code
    # Underflow below f32 normal range necessarily rounds to FP8 zero.
    exponent = fx.max(fx.Int32(1), 132 - difference)
    factor = (exponent.to(fx.Uint32) << 23).bitcast(fx.Float32)
    lo = _pack_fp8(raw, 0, factor)
    hi = _pack_fp8(raw, 16, factor)
    return fx.Vector.from_elements([lo, hi], fx.Int32).bitcast(fx.Float8E4M3FNUZ)


def _make_a8_loader(lds_ptr, *, arg_a, rows, row_base, bm, bk, stages, swizzle):
    from ..mxfp4_gemm_common import _global_i32_buffer_view

    tid = fx.Int32(gpu.thread_id("x"))
    lane = tid % 64
    width = 4 if bm * bk >= 4096 else 2
    copies = bm * bk // (256 * width * 4)
    global_view = _global_i32_buffer_view(arg_a, rows)
    tiles = fx.logical_divide(global_view, fx.make_layout(width, 1))
    src_atom = fx.make_copy_atom(rocdl.BufferCopy128b(2) if width == 4 else rocdl.BufferCopy64b(2), fx.Int32)
    lds_atom = fx.make_copy_atom(fx.UniversalCopy128b() if width == 4 else fx.UniversalCopy64b(), fx.Int32)
    read_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
    shared = fx.make_view(fx.recast_iter(fx.Int32, lds_ptr), fx.make_layout(stages * bm * bk // 4, 1))
    dst_tiles = fx.logical_divide(shared, fx.make_layout(width, 1))
    read_tiles = fx.logical_divide(shared, fx.make_layout(2, 1))
    coordinates = []
    for j in range_constexpr(copies):
        index = (tid + j * 256) * (width * 4)
        row, col = index // bk, index % bk
        physical = col ^ ((row & (bk // 16 - 1)) * 16) if swizzle else col
        coordinates.append((row_base(row), row * bk + physical, col))

    def store(base_k, slot):
        for base, physical, col in coordinates:
            r = fx.make_rmem_tensor(fx.make_layout(width, 1), fx.Int32)
            fx.copy(src_atom, fx.slice(tiles, (None, (base + base_k + col) // (width * 4))), r)
            fx.copy(lds_atom, r, fx.slice(dst_tiles, (None, (slot * bm * bk + physical) // (width * 4))))

    def load(mi, ku, slot):
        row = lane % 16 + mi * 16
        col = (lane // 16) * 32 + (ku // 4) * 128 + (ku % 4) * 8
        if swizzle:
            col = col ^ ((row & (bk // 16 - 1)) * 16)
        offset = slot * bm * bk + row * bk + col
        r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
        fx.copy(read_atom, fx.slice(read_tiles, (None, offset // 8)), r)
        return fx.Vector(r.load()).bitcast(fx.Float8E4M3FNUZ)

    return store, load


def validate_config(config=None):
    c = dict(bm=64, gate_n=128, gate_k=128, down_n=256, down_k=256,
             epilog="reduce", fast_decode=True, pipeline=False, gate_xcd=0, down_xcd=1, waves=0, flush_tiny=False,
             lazy_a=False, gate_bm=0, gate_swizzle=False, b_cache=2, port=False, k_loop="runtime",
             # port-only: gemm1 slice-K, per-stage waves/b_cache overrides (-1 = inherit),
             # gemm2 persistent CTA count (0 = one CTA per tile).
             k_wave=1, gate_waves=-1, down_waves=-1, gate_b_cache=-1, down_b_cache=-1, down_persist=0)
    if config:
        if set(config) - set(c):
            raise ValueError("unknown A8W4 configuration fields")
        c.update(config)
    for key, allowed in dict(bm=(16, 32, 64, 128, 256), gate_bm=(0, 16, 32, 64, 128), gate_n=(64, 128, 256), gate_k=(128, 256),
                             down_n=(64, 128, 256), down_k=(128, 256), gate_xcd=(0, 1, 2, 4, 8),
                             down_xcd=(0, 1, 2, 4, 8), waves=(0, 1, 2, 3, 4)).items():
        if type(c[key]) is not int or c[key] not in allowed:
            raise ValueError(f"invalid {key}={c[key]}")
    if type(c["b_cache"]) is not int or c["b_cache"] not in (0, 1, 2, 3):
        raise ValueError("b_cache must be a cache modifier in 0..3")
    if c["epilog"] not in ("atomic", "reduce"):
        raise ValueError("epilog must be atomic or reduce")
    if c["k_loop"] not in ("unroll", "runtime"):
        raise ValueError("k_loop must be unroll or runtime (port only)")
    for name in ("fast_decode", "pipeline", "flush_tiny", "lazy_a", "gate_swizzle", "port"):
        if type(c[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if c["port"] and (c["lazy_a"] or not c["fast_decode"] or c["gate_bm"] not in (0, c["bm"])):
        # The a16w4 port (a_dtype="fp8r") has neither lazy A nor the slow decoder,
        # and its stage1 tile_m is the sorting block_m; it always pipelines K > 1 tile.
        raise ValueError("port=True needs lazy_a=False, fast_decode=True, gate_bm in (0, bm)")
    if c["flush_tiny"] and not c["fast_decode"]:
        raise ValueError("flush_tiny is only supported by the lookup decoder")
    gate_bm = c["gate_bm"] or c["bm"]
    if gate_bm > c["bm"] or c["bm"] % gate_bm:
        raise ValueError("gate_bm must divide the sorting/down block-M")
    gate_lds = gate_bm * c["gate_k"] * (2 if c["pipeline"] else 1)
    down_lds = max(c["bm"] * c["down_k"], c["bm"] * c["down_n"] * (2 if c["epilog"] == "reduce" else 4))
    if max(gate_lds, down_lds) > 65536:
        raise ValueError("A8W4 configuration exceeds gfx942 64KiB LDS")
    return c


@cache
def _compile_gemm(stage, hidden, inter, experts, topk, bm, bn, bk, epilog, fast_decode, pipeline, xcd, waves, flush_tiny, lazy_a, sorting_bm, gate_swizzle, b_cache):
    from ..act import silu_mul_batch
    from .gemm2 import _atomic_bf16_epilog, _reduce_bf16_epilog
    from .utils import _global_i32_at, make_b_loader
    from ..mxfp4_gemm_common import _global_i32_buffer_tiles

    k = hidden if stage == 1 else inter
    n = inter if stage == 1 else hidden
    n_weight = 2 * inter if stage == 1 else hidden
    if k % bk or n % bn or n_weight % 256:
        raise ValueError("A8W4 requires whole K/N tiles and N-weight multiples of 256")
    m_reps, n_reps, k_steps = bm // 16, bn // 64, bk // 32
    k_tiles, n_blocks = k // bk, n // bn
    stages = 2 if stage == 1 and pipeline and k_tiles > 1 else 1
    if stages == 2 and k_tiles % 2:
        raise ValueError("two-slot pipeline requires an even K tile count")
    shared_bytes = stages * bm * bk
    if stage == 2:
        shared_bytes = max(shared_bytes, bm * bn * (2 if epilog == "reduce" else 4))
    if shared_bytes > 65536:
        raise ValueError("A8W4 LDS overflow")
    name = f"a8w4_g{stage}_h{hidden}_i{inter}_e{experts}_bm{bm}_sort{sorting_bm}_bn{bn}_bk{bk}_{epilog}_lut{int(fast_decode)}_p{int(pipeline)}_x{xcd}_w{waves}_ftz{int(flush_tiny)}_lazy{int(lazy_a)}_sw{int(gate_swizzle)}_bc{b_cache}"

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, shared_bytes, 16]

    @flyc.kernel(name=name, known_block_size=(256, 1, 1))
    def kernel(a: fx.Int64, a_scale: fx.Int64, b: fx.Int64, b_scale: fx.Int64, b_exp: fx.Int64,
               eids: fx.Int64, valid: fx.Int64, ids: fx.Int64, weights: fx.Int64,
               output: fx.Int64, tokens: fx.Int32):
        lds = fx.SharedAllocator().allocate(Shared).peek().raw.ptr
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % 64
        wave = fx.Int32(rocdl.readfirstlane(T.i32, (tid // 64).ir_value()))
        count = fx.Int32(_global_i32_at(valid, fx.Int32(0)))
        blocks = count // bm
        bound = blocks * n_blocks
        pid = fx.Int32(gpu.block_id("x"))
        if pid < bound:
            tile = pid
            if const_expr(xcd > 0):
                q, r = bound // 8, bound % 8
                tile = (pid % 8) * q + fx.min(pid % 8, r) + pid // 8
                group = tile // (xcd * n_blocks)
                first_m = group * xcd
                size_m = fx.min(blocks - first_m, xcd)
                local = tile % (xcd * n_blocks)
                tile = (first_m + local % size_m) * n_blocks + local // size_m
            m_block, n_block = tile // n_blocks, tile % n_blocks
            start_m, start_n = m_block * bm, n_block * bn
            expert = fx.Int32(rocdl.readfirstlane(T.i32, _global_i32_at(eids, m_block // (sorting_bm // bm)).ir_value()))
            loader = make_b_loader(b, b_scale, N_OUT=n_weight, K=k, NE=experts, e=expert,
                                   lane_div_16=lane // 16, lane_mod_16=lane % 16, TILE_K=bk,
                                   w_dtype="fp4", b_cache_mod=b_cache, rocm_arch="gfx942")
            cols = [loader.col(start_n, wave * (bn // 4), fx.Int32(ni * 16)) for ni in range_constexpr(n_reps)]
            if const_expr(stage == 1):
                up_cols = [loader.col(start_n, wave * (bn // 4), fx.Int32(ni * 16 + inter)) for ni in range_constexpr(n_reps)]
            exponents = []
            for ni in range_constexpr(n_reps):
                col = start_n + wave * (bn // 4) + ni * 16 + lane % 16
                exponents.append(fx.ptr_load(fx.add_offset(_pointer(b_exp, fx.Int32, 4), expert * n_weight + col)))
            if const_expr(stage == 1):
                up_exponents = []
                for ni in range_constexpr(n_reps):
                    col = start_n + wave * (bn // 4) + ni * 16 + lane % 16 + inter
                    up_exponents.append(fx.ptr_load(fx.add_offset(_pointer(b_exp, fx.Int32, 4), expert * n_weight + col)))

            def row_base(row):
                if const_expr(stage == 1):
                    packed_id = fx.Int32(_global_i32_at(ids, start_m + row))
                    return (packed_id & 0x00FFFFFF) * k
                else:
                    return (start_m + row) * k

            a_bytes = fx.Int64(tokens) * k if stage == 1 else fx.Int64(count) * k
            store_a, load_a = _make_a8_loader(lds, arg_a=a, rows=a_bytes, row_base=row_base,
                                             bm=bm, bk=bk, stages=stages, swizzle=stage == 2 or gate_swizzle)
            acc = [[fx.make_rmem_tensor(4, fx.Float32) for _ in range(n_reps)] for _ in range(m_reps)]
            for mi in range_constexpr(m_reps):
                for ni in range_constexpr(n_reps):
                    acc[mi][ni].fill(0.0)
            if const_expr(stage == 1):
                up = [[fx.make_rmem_tensor(4, fx.Float32) for _ in range(n_reps)] for _ in range(m_reps)]
                for mi in range_constexpr(m_reps):
                    for ni in range_constexpr(n_reps):
                        up[mi][ni].fill(0.0)
            mma = fx.make_mma_atom(rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))

            def load_b(base):
                raw = [loader.load_raw(base, c) for c in cols]
                scales = [loader.load_scale(base, c) for c in cols]
                if const_expr(stage == 1):
                    return (raw, scales, [loader.load_raw(base, c) for c in up_cols],
                            [loader.load_scale(base, c) for c in up_cols])
                return raw, scales

            def compute(bt, af, slot=0):
                for ni in range_constexpr(n_reps):
                    tables = [_fp8_tables(bt[1][ni][i * 4], exponents[ni], flush_tiny) for i in range_constexpr(bk // 128)] if fast_decode else [None] * (bk // 128)
                    if const_expr(stage == 1):
                        up_tables = [_fp8_tables(bt[3][ni][i * 4], up_exponents[ni], flush_tiny) for i in range_constexpr(bk // 128)] if fast_decode else [None] * (bk // 128)
                    for ku in range_constexpr(k_steps):
                        bf = _decode_fp8(bt[0][ni][ku // 4][ku % 4], bt[1][ni][ku], exponents[ni], fast_decode, tables[ku // 4])
                        if const_expr(stage == 1):
                            uf = _decode_fp8(bt[2][ni][ku // 4][ku % 4], bt[3][ni][ku], up_exponents[ni], fast_decode, up_tables[ku // 4])
                        for mi in range_constexpr(m_reps):
                            av = load_a(mi, ku, slot) if lazy_a else af[mi][ku]
                            _mma_fp8(mma, acc[mi][ni], av, bf)
                            if const_expr(stage == 1):
                                _mma_fp8(mma, up[mi][ni], av, uf)

            if const_expr(stages == 2):
                for pair in range(k_tiles // 2):
                    base = fx.Int32(pair) * (2 * bk)
                    store_a(base, 0)
                    current_b = load_b(base)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    gpu.barrier()
                    af = None if lazy_a else [[load_a(mi, ku, 0) for ku in range_constexpr(k_steps)] for mi in range_constexpr(m_reps)]
                    store_a(base + bk, 1)
                    next_b = load_b(base + bk)
                    compute(current_b, af)
                    rocdl.s_waitcnt(lgkmcnt=0)
                    gpu.barrier()
                    af = None if lazy_a else [[load_a(mi, ku, 1) for ku in range_constexpr(k_steps)] for mi in range_constexpr(m_reps)]
                    compute(next_b, af, 1)
                    gpu.barrier()
            else:
                for kt in range(k_tiles):
                    base = fx.Int32(kt) * bk
                    store_a(base, 0)
                    current_b = load_b(base)
                    gpu.barrier()
                    af = None if lazy_a else [[load_a(mi, ku, 0) for ku in range_constexpr(k_steps)] for mi in range_constexpr(m_reps)]
                    compute(current_b, af)
                    gpu.barrier()
            gpu.barrier()

            # Row and channel scales apply after K accumulation, before SiLU or routing weights.
            scale_tiles = _global_i32_buffer_tiles(a_scale, fx.Int64(tokens if stage == 1 else count) * 4, 1)
            scale_atom = fx.make_copy_atom(rocdl.BufferCopy32b(), fx.Int32)
            for mi in range_constexpr(m_reps):
                row_scales = []
                for ii in range_constexpr(4):
                    row = start_m + mi * 16 + (lane // 16) * 4 + ii
                    source_row = (fx.Int32(_global_i32_at(ids, row)) & 0x00FFFFFF) if stage == 1 else row
                    fragment = fx.make_rmem_tensor(1, fx.Int32)
                    fx.copy(scale_atom, fx.slice(scale_tiles, (None, source_row)), fragment)
                    row_scales.append(fx.Vector(fragment.load())[0].bitcast(fx.Float32))
                for ni in range_constexpr(n_reps):
                    factor = _channel_factor(exponents[ni])
                    value = fx.Vector(acc[mi][ni].load())
                    scaled = [value[ii] * (row_scales[ii] * factor) for ii in range_constexpr(4)]
                    if const_expr(stage == 1):
                        uf = _channel_factor(up_exponents[ni])
                        uv = fx.Vector(up[mi][ni].load())
                        col = start_n + wave * (bn // 4) + ni * 16 + lane % 16
                        for ii in range_constexpr(4):
                            g = scaled[ii]
                            u = uv[ii] * (row_scales[ii] * uf)
                            y = silu_mul_batch([g], [u])[0]
                            row = start_m + mi * 16 + (lane // 16) * 4 + ii
                            # Including padding rows: a is zero there, so this writes
                            # zero and gives the following quantizer initialized input.
                            fx.ptr_store(y.to(fx.BFloat16), fx.add_offset(_pointer(output, fx.BFloat16, 2),
                                                                       fx.Int64(row) * inter + col))
                    else:
                        acc[mi][ni].store(fx.Vector.from_elements(scaled, fx.Float32))
            if const_expr(stage == 2):
                vectors = [[acc[mi][ni].load().ir_value() for ni in range(n_reps)] for mi in range(m_reps)]
                lds_address = fx.Int32(fx.ptrtoint(lds))
                if const_expr(epilog == "reduce"):
                    _reduce_bf16_epilog(lds_address, vectors, output, ids, weights, start_m, n_block,
                                        wave, lane, tokens, bm, hidden, bn, topk)
                else:
                    _atomic_bf16_epilog(lds_address, vectors, output, ids, weights, start_m, n_block,
                                        wave, lane, tokens, bm, hidden, bn, True)

    @flyc.jit
    def launch(a: fx.Int64, a_scale: fx.Int64, b: fx.Int64, b_scale: fx.Int64, b_exp: fx.Int64,
               eids: fx.Int64, valid: fx.Int64, ids: fx.Int64, weights: fx.Int64,
               output: fx.Int64, tokens: fx.Int32, grid: fx.Int32, stream: fx.Stream):
        kernel(a, a_scale, b, b_scale, b_exp, eids, valid, ids, weights, output, tokens,
               value_attrs={"rocdl.waves_per_eu": waves} if waves else None).launch(
                   grid=(fx.Int64(grid),), block=(256,), stream=stream)

    return launch


class A8W4MoE:
    """Experimental runner alongside DynamicMoE, not a default replacement.

    Caller supplies standard-preshuffled E2M1/E8M0 tensors and in-range unique
    expert IDs per token (0 <= id < E). Values/scales must produce finite
    intermediate activations; this is not a promise for every BF16 exponent.
    No CPU value scan/synchronization or content-dependent cache is in infer.
    flush_tiny defaults OFF and is an explicit extra approximation in the
    tuned profile; FP8 activation quantization remains lossy in both modes.
    """

    def __init__(self, x, ids, weights, packed, output, *, config=None):
        if x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_cuda or not x.is_contiguous():
            raise ValueError("A8W4 requires a contiguous CUDA BF16 [tokens,hidden] input")
        if x.shape[0] < 1 or x.shape[0] >= 2**24 or x.numel() >= 2**31:
            raise ValueError("A8W4 requires positive token count and 32-bit-safe input offsets")
        if ids.ndim != 2 or ids.shape[0] != x.shape[0] or not 1 <= ids.shape[1] < 256:
            raise ValueError("invalid top-k ID shape")
        if (ids.dtype != torch.int32 or weights.dtype != torch.float32 or weights.shape != ids.shape
                or output.dtype != x.dtype or output.shape != x.shape):
            raise ValueError("IDs/route weights/output have invalid dtype or shape")
        tensors = (ids, weights, output, packed.w1, packed.s1, packed.w2, packed.s2)
        if any(t.device != x.device or not t.is_contiguous() for t in tensors):
            raise ValueError("all A8W4 tensors must be contiguous on the input device")
        #if packed.w1.dtype != torch.uint8 or packed.w2.dtype != torch.uint8:
        #    raise ValueError("A8W4 weights must be packed MXFP4 uint8")
        self.x, self.ids, self.weights, self.packed, self.output = x, ids, weights, packed, output
        self.config = validate_config(config)
        self.hidden, self.inter, self.experts, self.topk = x.shape[1], packed.w1.shape[1] // 2, packed.w1.shape[0], ids.shape[1]
        if self.hidden % 256 or self.inter % 256 or self.experts < 1:
            raise ValueError("A8W4 requires hidden/inter multiples of 256 and positive experts")
        if (packed.w1.shape != (self.experts, 2 * self.inter, self.hidden // 2)
                or packed.w2.shape != (self.experts, self.hidden, self.inter // 2)
                or packed.s1.dtype != torch.float8_e8m0fnu or packed.s2.dtype != torch.float8_e8m0fnu
                or packed.s1.numel() != self.experts * 2 * self.inter * (self.hidden // 32)
                or packed.s2.numel() != self.experts * self.hidden * (self.inter // 32)):
            raise ValueError(f"packed MXFP4 weight/scale shapes {packed.w1.shape}{packed.s1.dtype} {packed.w2.shape}{packed.s2.dtype} do not match the MoE {self.experts, self.hidden, self.inter}")
        if max(packed.w1.nbytes, packed.w2.nbytes) >= 2**31:
            raise ValueError("prototype weight offsets must fit signed 32 bits")
        if x.shape[0] * self.topk * self.hidden >= 2**31:
            raise ValueError("prototype routed-output offsets must fit signed 32 bits")
        for tensor in (x, ids, weights, packed.w1, packed.s1, packed.w2, packed.s2):
            if output.data_ptr() < tensor.data_ptr() + tensor.nbytes and tensor.data_ptr() < output.data_ptr() + output.nbytes:
                raise ValueError("output must not overlap A8W4 inputs")
        if not torch.cuda.get_device_properties(x.device).gcnArchName.startswith("gfx942"):
            raise ValueError("A8W4 prototype requires gfx942 FP8 MFMA")
        c = self.config
        self.gate_bm = c["gate_bm"] or c["bm"]
        if c["port"]:
            from .gemm1 import compile_gemm1_a16w4_port
            from .gemm2 import compile_gemm2_a16w4_port

            waves = c["waves"] or None
            gw = waves if c["gate_waves"] < 0 else (c["gate_waves"] or None)
            dw = waves if c["down_waves"] < 0 else (c["down_waves"] or None)
            gbc = c["b_cache"] if c["gate_b_cache"] < 0 else c["gate_b_cache"]
            dbc = c["b_cache"] if c["down_b_cache"] < 0 else c["down_b_cache"]
            self.g1 = compile_gemm1_a16w4_port(
                c["bm"], D_HIDDEN=self.hidden, D_INTER=self.inter, NE=self.experts, TOPK=self.topk,
                TILE_N=c["gate_n"], TILE_K=c["gate_k"], b_cache_mod=gbc, xcd_swizzle=c["gate_xcd"],
                waves_per_eu=gw, a_dtype="fp8r", a_swizzle=c["gate_swizzle"], flush_tiny=c["flush_tiny"],
                k_loop=c["k_loop"], k_wave=c["k_wave"], rocm_arch="gfx942")
            self.g2 = compile_gemm2_a16w4_port(
                c["bm"], NE=self.experts, N_OUT=self.hidden, D_INTER=self.inter, TILE_N=c["down_n"],
                TILE_K=c["down_k"], xcd_swizzle=c["down_xcd"], b_cache_mod=dbc, waves_per_eu=dw,
                a_dtype="fp8r", flush_tiny=c["flush_tiny"], persist=c["down_persist"] > 0,
                epilog=c["epilog"], topk=self.topk, rocm_arch="gfx942")
        else:
            self.g1 = _compile_gemm(1, self.hidden, self.inter, self.experts, self.topk, self.gate_bm, c["gate_n"],
                                    c["gate_k"], c["epilog"], c["fast_decode"], c["pipeline"], c["gate_xcd"], c["waves"], c["flush_tiny"], c["lazy_a"], c["bm"], c["gate_swizzle"], c["b_cache"])
            self.g2 = _compile_gemm(2, self.hidden, self.inter, self.experts, self.topk, c["bm"], c["down_n"],
                                    c["down_k"], c["epilog"], c["fast_decode"], False, c["down_xcd"], c["waves"], c["flush_tiny"], c["lazy_a"], c["bm"], c["gate_swizzle"], c["b_cache"])
        self.reducer = None
        assert c["port"] or c["epilog"] == "atomic"
        #from ..moe_gemm_2stage.gfx9_fp8.moe_reduce import compile_moe_reduction
        #self.reducer = compile_moe_reduction(topk=self.topk, model_dim=self.hidden, dtype_str="bf16")

    def __call__(self):
        # Lazy: moe_kernels imports this module, and aiter.fused_moe imports moe_kernels.
        from aiter.fused_moe import moe_sorting

        c, p = self.config, self.packed
        # Port reduce epilog: stage2 writes unique routing-weighted [T, topk, H] rows,
        # then aiter's moe reduction sums over topk (no atomics, no zeroed output).
        use_reduce = self.reducer is not None or (c["port"] and c["epilog"] == "reduce")
        si, sw, se, valid, _ = moe_sorting(self.ids, self.weights, self.experts, self.hidden, self.x.dtype,
                                         c["bm"], accumulate=not use_reduce,
                                         output=None if use_reduce else self.output)[:5]
        ex1, ex2 = channel_exponents(p.w1, p.s1), channel_exponents(p.w2, p.s2)
        a1, as1 = quantize_rows(self.x)
        mid = torch.empty((se.numel() * c["bm"], self.inter), device=self.x.device, dtype=torch.bfloat16)
        # Kept for validation (stage1 output is atomic-free, so comparable bit-wise).
        self.mid, self.sorted_ids, self.num_valid = mid, si, valid
        stream = torch.cuda.current_stream()
        tokens = self.x.shape[0]
        grid1 = se.numel() * (c["bm"] // self.gate_bm) * (self.inter // c["gate_n"])
        grid2 = se.numel() * (self.hidden // c["down_n"])
        if c["port"]:
            if c["down_persist"] > 0:
                grid2 = min(grid2, c["down_persist"])
            # a16w4 port argument order; gemm1 takes a_scale / b_exp as its two trailing
            # fp8r args, gemm2 only a_scale (it derives the W2 exponent in-kernel).
            args1 = (a1.data_ptr(), p.w1.data_ptr(), p.s1.data_ptr(), se.data_ptr(), valid.data_ptr(),
                     si.data_ptr(), tokens, grid1, 0.0, 0.0, 0.0, 0.0, 0.0, mid.data_ptr(),
                     as1.data_ptr(), ex1.data_ptr(), stream)
            _run_compiled(self.g1, *args1)
            a2, as2 = quantize_rows(mid, valid)
            target = (torch.empty((tokens, self.topk, self.hidden), device=self.x.device, dtype=self.x.dtype)
                      if use_reduce else self.output)
            args2 = (a2.data_ptr(), p.w2.data_ptr(), p.s2.data_ptr(), se.data_ptr(), valid.data_ptr(),
                     si.data_ptr(), sw.data_ptr(), tokens, se.numel(), grid2, target.data_ptr(),
                     as2.data_ptr(), stream)
            _run_compiled(self.g2, *args2)
            if use_reduce:
                from ...moe_kernels import _run_moe_reduction

                _run_moe_reduction(target, self.output, tokens, self.topk, self.hidden, stream=stream)
            # Per-stage replay hooks for timing (buffers stay alive via the closure).
            self._keep = (si, sw, se, valid, ex1, ex2, a1, as1, a2, as2, mid, target)
            self.stage1 = lambda: _run_compiled(self.g1, *args1)
            if use_reduce:
                def _stage2():
                    _run_compiled(self.g2, *args2)
                    _run_moe_reduction(target, self.output, tokens, self.topk, self.hidden, stream=stream)

                self.stage2 = _stage2
            else:
                self.stage2 = lambda: _run_compiled(self.g2, *args2)
            return self.output
        args1 = (a1.data_ptr(), as1.data_ptr(), p.w1.data_ptr(), p.s1.data_ptr(), ex1.data_ptr(),
                 se.data_ptr(), valid.data_ptr(), si.data_ptr(), sw.data_ptr(), mid.data_ptr(),
                 tokens, grid1, stream)
        _run_compiled(self.g1, *args1)
        a2, as2 = quantize_rows(mid, valid)
        route_output = (torch.empty((self.x.shape[0], self.topk, self.hidden), device=self.x.device, dtype=self.x.dtype)
                        if use_reduce else self.output)
        args2 = (a2.data_ptr(), as2.data_ptr(), p.w2.data_ptr(), p.s2.data_ptr(), ex2.data_ptr(),
                 se.data_ptr(), valid.data_ptr(), si.data_ptr(), sw.data_ptr(), route_output.data_ptr(),
                 self.x.shape[0], se.numel() * (self.hidden // c["down_n"]), stream)
        _run_compiled(self.g2, *args2)
        self._keep = (si, sw, se, valid, ex1, ex2, a1, as1, a2, as2, mid)
        self.stage1 = lambda: _run_compiled(self.g1, *args1)
        self.stage2 = lambda: _run_compiled(self.g2, *args2)
        
        #if use_reduce:
        #    _run_compiled(self.reducer, flyc.from_c_void_p(fx.Uint8, route_output.data_ptr()),
        #                  flyc.from_c_void_p(fx.Uint8, self.output.data_ptr()), flyc.from_c_void_p(fx.Uint8, 0),
        #                  flyc.from_c_void_p(fx.Uint8, self.ids.data_ptr()), self.x.shape[0], stream)
        return route_output
