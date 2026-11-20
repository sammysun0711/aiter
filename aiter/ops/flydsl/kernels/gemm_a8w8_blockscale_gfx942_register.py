# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Transposed FNUZ MFMA with wave-local K128 activation prefetch on CDNA3."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec


def compile_register_gemm(n, k, symbol):
    # Four waves span M. Each wave owns 32 rows by 128 columns.
    rm, rn = 2, 8
    groups = k // 128

    @flyc.jit
    def load_pair(ptr, word):
        source = fx.make_view(ptr + fx.Int32(word), fx.make_layout(2, 1))
        frag = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int64)
        atom = fx.make_copy_atom(fx.rocdl.BufferCopy(128), fx.Int64)
        fx.copy_atom_call(atom, source, frag)
        return frag.load()

    @flyc.jit
    def read_a(ptr, row, lane, g):
        parts = []
        for pair in range_constexpr(2):
            for mi in range_constexpr(rm):
                r = (row + mi * 16 + lane % 16) & 127
                word = (
                    fx.Int64(r) * (k // 8)
                    + fx.Int64(g) * 16
                    + pair * 8
                    + 2 * (lane // 16)
                )
                v = Vec(load_pair(ptr, word))
                parts.extend([v[0], v[1]])
        return Vec.from_elements(parts, fx.Int64)

    @flyc.jit
    def accumulate(bp, asp, bsp, row, col, pn, lane, rows, g, running, cached_a):
        atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
        av = Vec(cached_a)
        bparts = []
        for pair in range_constexpr(2):
            for ni in range_constexpr(rn):
                c = (col + ni * 16 + lane % 16) & 127
                word = (
                    fx.Int64(c // 16) * (2 * k)
                    + fx.Int64(g) * 256
                    + pair * 128
                    + (lane // 16) * 32
                    + (lane % 16) * 2
                )
                v = Vec(load_pair(bp, word))
                bparts.extend([v[0], v[1]])
        bv = Vec.from_elements(bparts, fx.Int64)
        bscale = fx.Float32(fx.ptr_load(bsp + fx.Int64(pn) * groups + fx.Int64(g)))
        sa_group = asp + fx.Int64(g) * fx.Int64(rows)
        prev = Vec(running)
        updated = []
        for mi in range_constexpr(rm):
            r = row + mi * 16 + lane % 16
            scale = fx.Float32(fx.ptr_load(sa_group + fx.Int64(r))) * bscale
            for ni in range_constexpr(rn):
                partial = fx.make_rmem_tensor(4, fx.Float32)
                partial.store(fx.full(4, 0.0, fx.Float32))
                for sub in range_constexpr(4):
                    af = fx.make_rmem_tensor(1, fx.Int64)
                    bf = fx.make_rmem_tensor(1, fx.Int64)
                    af.store(
                        Vec.from_elements([av[((sub // 2) * rm + mi) * 2 + sub % 2]])
                    )
                    bf.store(
                        Vec.from_elements([bv[((sub // 2) * rn + ni) * 2 + sub % 2]])
                    )
                    # W @ X.T gives this lane one activation row and four adjacent output columns.
                    fx.gemm(atom, partial, bf, af, partial)
                vals = Vec(partial.load())
                for i in range_constexpr(4):
                    updated.append(
                        fx.math.fma(vals[i], scale, prev[(mi * rn + ni) * 4 + i])
                    )
        return Vec.from_elements(updated, fx.Float32)

    @flyc.kernel(name=symbol, known_block_size=[256, 1, 1])
    def kernel(
        a: fx.Tensor,
        b: fx.Tensor,
        out: fx.Tensor,
        sa: fx.Tensor,
        sb: fx.Tensor,
        rows: fx.Int32,
    ):
        pm, pn = fx.Int32(fx.block_idx.x), fx.Int32(fx.block_idx.y)
        tid = fx.Int32(fx.thread_idx.x)
        wave, lane = tid // 64, tid % 64
        row, col = pm * 128 + wave * 32, pn * 128
        # Buffer offsets are tile-relative. Rebase in i64 for matrices larger than 2 GiB.
        ap = fx.rocdl.make_buffer_ptr(
            fx.get_iter(a) + fx.Int64(pm) * (128 * k // 8), num_records_bytes=128 * k
        )
        bp = fx.rocdl.make_buffer_ptr(
            fx.get_iter(b) + fx.Int64(pn) * (128 * k // 8), num_records_bytes=128 * k
        )
        cp, asp, bsp = fx.get_iter(out), fx.get_iter(sa), fx.get_iter(sb)
        result = fx.full(4 * rm * rn, 0.0, fx.Float32)
        cur_a = read_a(ap, row, lane, fx.Int32(0))
        for iv, state in range(0, groups - 1, 1, init=[result, cur_a]):
            g = fx.Int32(iv)
            next_a = read_a(ap, row, lane, g + 1)
            step = accumulate(
                bp, asp, bsp, row, col, pn, lane, rows, g, state[0], state[1]
            )
            result, cur_a = yield [step, next_a]
        result = accumulate(
            bp, asp, bsp, row, col, pn, lane, rows, fx.Int32(groups - 1), result, cur_a
        )
        total = Vec(result)
        for mi in range_constexpr(rm):
            for ni in range_constexpr(rn):
                r = row + mi * 16 + lane % 16
                c = col + ni * 16 + (lane // 16) * 4
                values = [
                    total[(mi * rn + ni) * 4 + i].to(out.element_type)
                    for i in range_constexpr(4)
                ]
                fx.ptr_store(
                    Vec.from_elements(values), cp + fx.Int64(r) * n + fx.Int64(c)
                )

    @flyc.jit
    def launch(
        a: fx.Tensor,
        b: fx.Tensor,
        out: fx.Tensor,
        sa: fx.Tensor,
        sb: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(a, b, out, sa, sb, rows).launch(
            grid=(rows // 128, n // 128, 1), block=(256, 1, 1), stream=stream
        )

    return launch
