# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CDNA3 FNUZ MFMA with direct loads or two-stage LDS activation tiles."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec


def compile_blockscale_gfx942(n, k, tile_m, tile_n, preshuffle_b, variant=""):
    # Four wave64s arranged 2x2. Each 16x16 accumulator contains four rows per lane.
    rm, rn = tile_m // 32, tile_n // 32
    kb_count = k // 128
    elem_count = 4 * rm * rn
    symbol = f"flydsl_blockscale_gfx942_{tile_m}x{tile_n}x128_ps{int(preshuffle_b)}"
    staged = variant in ("lds_m", "lds_n")
    n_first = variant == "lds_n"
    if staged:
        symbol += f"_{variant}"
    stage_words = tile_m * 16  # 128 FP8 elements per row, eight per i64.

    @fx.struct
    class Shared:
        data: fx.Array[fx.Int64, 2 * stage_words, 16]

    @flyc.jit
    def stage_a(a8, lds, pm, g, slot, tid):
        pair = tid % 8
        for turn in range_constexpr(tile_m // 32):
            row = tid // 8 + turn * 32
            off = fx.Int64(pm * tile_m + row) * (k // 8) + fx.Int64(g) * 16 + pair * 2
            data = fx.ptr_load(a8 + off, result_type=fx.Vector.make_type(2, fx.Int64))
            dest = slot * stage_words + row * 16 + (pair ^ (row % 8)) * 2
            fx.ptr_store(data, lds + dest)

    @flyc.kernel(name=symbol, known_block_size=[256, 1, 1])
    def kernel(
        a: fx.Tensor,
        b: fx.Tensor,
        out: fx.Tensor,
        sa: fx.Tensor,
        sb: fx.Tensor,
        m: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane, wave = tid % 64, tid // 64
        if const_expr(n_first):
            pm, pn = fx.Int32(fx.block_idx.y), fx.Int32(fx.block_idx.x)
        else:
            pm, pn = fx.Int32(fx.block_idx.x), fx.Int32(fx.block_idx.y)
        row = pm * tile_m + (wave // 2) * (16 * rm)
        col = pn * tile_n + (wave % 2) * (16 * rn)
        a8 = fx.get_iter(a)
        b8 = fx.get_iter(b)
        ap, bp = fx.get_iter(sa), fx.get_iter(sb)
        cp = fx.get_iter(out)
        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
        result = fx.full(elem_count, 0.0, fx.Float32)
        if const_expr(staged):
            lds = fx.SharedAllocator(static=True).allocate(Shared).peek().data.ptr
            stage_a(a8, lds, pm, fx.Int32(0), fx.Int32(0), tid)
        for group, state in range(0, kb_count, 1, init=[result]):
            g = fx.Int32(group)
            if const_expr(staged):
                fx.gpu.barrier()
                if g + 1 < kb_count:
                    stage_a(a8, lds, pm, g + 1, (g + 1) % 2, tid)
            partial = [
                fx.make_rmem_tensor(4, fx.Float32) for _ in range_constexpr(rm * rn)
            ]
            for i in range_constexpr(rm * rn):
                partial[i].store(fx.full(4, 0.0, fx.Float32))
            for sub in range_constexpr(4):
                # Each operand contributes eight bytes per lane to each K32 MFMA.
                # Both operands use the same permutation within this K128 group.
                word = g * 16 + (sub // 2) * 8 + (lane // 16) * 2 + sub % 2
                av = [fx.make_rmem_tensor(1, fx.Int64) for _ in range_constexpr(rm)]
                bv = [fx.make_rmem_tensor(1, fx.Int64) for _ in range_constexpr(rn)]
                for mi in range_constexpr(rm):
                    if const_expr(staged):
                        r = (wave // 2) * 16 * rm + mi * 16 + lane % 16
                        pair = (sub // 2) * 4 + lane // 16
                        off = (
                            (g % 2) * stage_words
                            + r * 16
                            + (pair ^ (r % 8)) * 2
                            + sub % 2
                        )
                        value = fx.ptr_load(lds + off)
                    else:
                        off = fx.Int64(row + mi * 16 + lane % 16) * (k // 8) + fx.Int64(
                            word
                        )
                        value = fx.ptr_load(a8 + off)
                    av[mi].store(Vec.from_elements([fx.Int64(value)]))
                for ni in range_constexpr(rn):
                    channel = col + ni * 16 + lane % 16
                    if const_expr(preshuffle_b):
                        off = (
                            fx.Int64(channel // 16) * (2 * k)
                            + fx.Int64(word // 2) * 32
                            + (channel % 16) * 2
                            + sub % 2
                        )
                    else:
                        off = fx.Int64(channel) * (k // 8) + fx.Int64(word)
                    bv[ni].store(Vec.from_elements([fx.Int64(fx.ptr_load(b8 + off))]))
                for mi in range_constexpr(rm):
                    for ni in range_constexpr(rn):
                        frag = partial[mi * rn + ni]
                        fx.gemm(atom, frag, av[mi], bv[ni], frag)
            prev = Vec(state[0])
            values = []
            if const_expr(staged):
                bscale = fx.Float32(
                    fx.ptr_load(bp + fx.Int64(pn) * kb_count + fx.Int64(g))
                )
            for mi in range_constexpr(rm):
                row4 = row + mi * 16 + (lane // 16) * 4
                ascales = [
                    fx.Float32(
                        fx.ptr_load(ap + fx.Int64(g) * fx.Int64(m) + fx.Int64(row4 + i))
                    )
                    for i in range_constexpr(4)
                ]
                for ni in range_constexpr(rn):
                    if const_expr(not staged):
                        channel = col + ni * 16 + lane % 16
                        bscale = fx.Float32(
                            fx.ptr_load(
                                bp + fx.Int64(channel // 128) * kb_count + fx.Int64(g)
                            )
                        )
                    v = Vec(partial[mi * rn + ni].load())
                    for i in range_constexpr(4):
                        pos = (mi * rn + ni) * 4 + i
                        if const_expr(staged):
                            values.append(
                                fx.math.fma(v[i], ascales[i] * bscale, prev[pos])
                            )
                        else:
                            values.append(prev[pos] + v[i] * (ascales[i] * bscale))
            result = yield [Vec.from_elements(values, fx.Float32)]

        total = Vec(result)
        for mi in range_constexpr(rm):
            for ni in range_constexpr(rn):
                ccol = col + ni * 16 + lane % 16
                for i in range_constexpr(4):
                    crow = row + mi * 16 + (lane // 16) * 4 + i
                    off = fx.Int64(crow) * n + fx.Int64(ccol)
                    fx.ptr_store(
                        total[(mi * rn + ni) * 4 + i].to(out.element_type), cp + off
                    )

    @flyc.jit
    def launch(
        a: fx.Tensor,
        b: fx.Tensor,
        out: fx.Tensor,
        sa: fx.Tensor,
        sb: fx.Tensor,
        m: fx.Int32,
        stream: fx.Stream,
    ):
        if const_expr(n_first):
            grid = (n // tile_n, m // tile_m, 1)
        else:
            grid = (m // tile_m, n // tile_n, 1)
        kernel(a, b, out, sa, sb, m).launch(grid=grid, block=(256, 1, 1), stream=stream)

    return launch
