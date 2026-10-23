# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gate/Up GEMM + SiLU + group128 FP8 quantization in one GPU kernel.

Input X is already FP8. Gate/Up results stay in registers until SiLU,
multiplication and quantization produce the FP8 input for Down GEMM.
"""

import functools
import flydsl.compiler as fc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from .common import _buffer, _i32, _fragment, _weight_fragment, _dot, _maximum, _scale
from .configuration import WeightCache
from .common import PADDED_COUNT
from .common import _route_token, _expert_id, _expert_reuse

LOG2_E = 1.4426950408889634


@functools.lru_cache(None)
def gate_up_fused_fp8_quant_launcher(
    hidden,
    intermediate,
    topk,
    block_m=32,
    slice_n=64,
    cache_policy=WeightCache.STREAM,
    workers=0,
    parallel_n=False,
    reuse_metadata=False,
):
    """First GEMM: gate/up, SiLU and group128 FP8 quantization in one kernel.

    Each block computes one M tile by N128 task, using smaller N fragments.
    The launch grid uses the physical capacity; GPU counts bound live work."""
    reduction = hidden
    output_dim = intermediate
    weight_n = 2 * intermediate
    output_tile = 128
    rows_per_wave = min(32, block_m) if parallel_n else block_m
    waves = 2 * (block_m // rows_per_wave) if parallel_n else 1
    num_m = rows_per_wave // 16
    assert slice_n in (16, 32, 64) and block_m in (16, 32, 64)
    assert not parallel_n or slice_n == 64
    num_n = slice_n // 16

    @fx.struct
    class SharedPeaks:
        values: fx.Array[fx.Float32, 2 * block_m, 16]

    @fc.kernel(
        name=f"flydsl_moe_ep_decode_fp8_b128_gfx950_s1_m{block_m}n{output_tile}k128_subn{slice_n}_c{int(cache_policy)}_np{int(parallel_n)}",
        known_block_size=[64 * waves, 1, 1],
    )
    def kernel(
        A: fx.Tensor,
        W: fx.Tensor,
        SA: fx.Tensor,
        SW: fx.Tensor,
        IDS: fx.Tensor,
        EXPERTS: fx.Tensor,
        VALID: fx.Tensor,
        COUNT: fx.Tensor,
        AUX: fx.Tensor,
        OUT: fx.Tensor,
    ):
        thread = fx.Int32(fx.thread_idx.x)
        lane, wave = thread % 64, thread // 64
        wave_m, wave_n = (
            (wave // 2, wave % 2) if parallel_n else (fx.Int32(0), fx.Int32(0))
        )
        shared = (
            fx.SharedAllocator().allocate(SharedPeaks).peek().values.ptr
            if parallel_n
            else None
        )
        (row_lane, k_lane) = (lane % 16, lane // 16)
        (ar, wr, sar, swr) = (_buffer(A), _buffer(W), _buffer(SA), _buffer(SW))
        (ids, experts) = (_buffer(IDS), _buffer(EXPERTS))
        (aux, out) = (_buffer(AUX), _buffer(OUT))
        padded = _i32(_buffer(VALID), PADDED_COUNT, uniform=True)
        received = _i32(_buffer(COUNT), 0, uniform=True)
        ntiles = output_dim // output_tile

        def compute_task(task, expert, cache_tile):
            (mtile, nbase) = (task // ntiles, task % ntiles * output_tile)
            # Input fragments gather token rows; result fragments own different rows.
            source = [
                _route_token(
                    _i32(
                        ids,
                        mtile * block_m + wave_m * rows_per_wave + mi * 16 + row_lane,
                    )
                )
                for mi in range_constexpr(num_m)
            ]
            rows = [
                mtile * block_m + wave_m * rows_per_wave + mi * 16 + k_lane * 4 + i
                for mi in range_constexpr(num_m)
                for i in range_constexpr(4)
            ]
            routes = [_i32(ids, r) for r in rows]
            tokens = [_route_token(r) for r in routes]

            def load_activation(kb):
                aa, sa = [], []
                for mi in range_constexpr(num_m):
                    arow = source[mi]
                    aa.append(
                        _fragment(
                            ar,
                            arow * reduction + k_lane * 16,
                            64,
                            source[mi] < received,
                            soffset_bytes=kb * 128,
                        )
                    )
                    for i in range_constexpr(4):
                        r = tokens[mi * 4 + i]
                        sa.append(
                            fx.Float32(
                                bo.buffer_load(
                                    sar,
                                    r * (reduction // 128),
                                    soffset_bytes=kb * (1) * 4,
                                    vec_width=1,
                                    dtype=fx.Float32,
                                    mask=tokens[mi * 4 + i] < received,
                                )
                            )
                        )
                return (
                    Vec.from_elements(
                        [a[i] for a in aa for i in range_constexpr(8)], fx.Int32
                    ),
                    Vec.from_elements(sa, fx.Float32),
                )

            def load(kb, n_offset):
                sw = _scale(
                    swr,
                    (expert * (weight_n // 128) + (nbase + n_offset) // 128)
                    * (reduction // 128)
                    + kb,
                )
                su = _scale(
                    swr,
                    (
                        expert * (weight_n // 128)
                        + (nbase + n_offset + intermediate) // 128
                    )
                    * (reduction // 128)
                    + kb,
                )
                av, sa = load_activation(kb)
                bb, uu = [], []
                for ni in range_constexpr(num_n):
                    scalar_base = expert * weight_n * reduction
                    scalar_base += (
                        (nbase + n_offset) // 16 + ni
                    ) * reduction * 16 + kb * 2048
                    lane_offset = row_lane * 16 + k_lane * 256
                    bb.append(
                        _weight_fragment(
                            wr,
                            lane_offset,
                            1024,
                            scalar_base=scalar_base,
                            cached=cache_tile,
                            adaptive=cache_policy == WeightCache.ADAPTIVE,
                        )
                    )
                    uu.append(
                        _weight_fragment(
                            wr,
                            lane_offset,
                            1024,
                            scalar_base=scalar_base + intermediate * reduction,
                            cached=cache_tile,
                            adaptive=cache_policy == WeightCache.ADAPTIVE,
                        )
                    )
                bv = Vec.from_elements(
                    [b[i] for b in bb for i in range_constexpr(8)], fx.Int32
                )
                uv = Vec.from_elements(
                    [u[i] for u in uu for i in range_constexpr(8)], fx.Int32
                )
                return (av, bv, uv, sa, sw, su)

            def step(g, u, av, bv, uv, sa, sw, su):
                (gv, uv_out) = ([], [])
                for mi in range_constexpr(num_m):
                    a = Vec.from_elements(
                        [av[mi * 8 + i] for i in range_constexpr(8)], fx.Int32
                    )
                    scale = Vec.from_elements(
                        [sa[mi * 4 + i] for i in range_constexpr(4)], fx.Float32
                    )
                    gate_scale, up_scale = scale * sw, scale * su
                    for ni in range_constexpr(num_n):
                        off = (mi * num_n + ni) * 4
                        b = Vec.from_elements(
                            [bv[ni * 8 + i] for i in range_constexpr(8)], fx.Int32
                        )
                        old = Vec.from_elements(
                            [g[off + i] for i in range_constexpr(4)], fx.Float32
                        )
                        result = old + _dot(a, b) * gate_scale
                        gv.extend([result[i] for i in range_constexpr(4)])
                        b_up = Vec.from_elements(
                            [uv[ni * 8 + i] for i in range_constexpr(8)], fx.Int32
                        )
                        old_up = Vec.from_elements(
                            [u[off + i] for i in range_constexpr(4)], fx.Float32
                        )
                        result_up = old_up + _dot(a, b_up) * up_scale
                        uv_out.extend([result_up[i] for i in range_constexpr(4)])
                return (
                    Vec.from_elements(gv, fx.Float32),
                    Vec.from_elements(uv_out, fx.Float32),
                )

            def gemm(n_offset):
                # N is uniform within each wave, including the two N64 waves.
                # Keep its base in SGPRs while lane offsets remain in VGPRs.
                n_offset = fx.Int32(
                    rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(n_offset))
                )
                (g, u) = (
                    fx.full(num_m * num_n * 4, 0.0, fx.Float32),
                    fx.full(num_m * num_n * 4, 0.0, fx.Float32),
                )
                (av, bv, uv, sa, sw, su) = load(fx.Int32(0), n_offset)
                for kb in range(reduction // 128 - 1):
                    (an, bn, un, san, swn, sun) = load(kb + 1, n_offset)
                    (g, u) = step(g, u, av, bv, uv, sa, sw, su)
                    (av, bv, uv, sa, sw, su) = (an, bn, un, san, swn, sun)
                return step(g, u, av, bv, uv, sa, sw, su)

            if parallel_n:
                # Each N wave computes 64 channels. Exchange only row maxima
                # once, after GEMM, to preserve the 128-channel FP8 scale group.
                g, u = gemm(wave_n * 64)
                z = g / (1.0 + fx.exp2(-g * LOG2_E)) * u
                local_peaks = []
                for mi in range_constexpr(num_m):
                    for i in range_constexpr(4):
                        oi = mi * 4 + i
                        peak = fx.Float32(0.0)
                        for ni in range_constexpr(4):
                            value = z[(mi * 4 + ni) * 4 + i]
                            peak = _maximum(peak, _maximum(value, -value))
                        for shift in range_constexpr(4):
                            peak = _maximum(
                                peak,
                                peak.shuffle_xor(fx.Int32(1 << shift), fx.Int32(16)),
                            )
                        local_peaks.append(peak)
                        if row_lane == 0:
                            offset = wave_n * block_m + rows[oi] - mtile * block_m
                            fx.ptr_store(peak, shared + fx.Int64(offset))
                fx.gpu.barrier()
                for mi in range_constexpr(num_m):
                    for i in range_constexpr(4):
                        oi = mi * 4 + i
                        other = fx.ptr_load(
                            shared
                            + fx.Int64(
                                (1 - wave_n) * block_m + rows[oi] - mtile * block_m
                            )
                        )
                        scale = (
                            _maximum(
                                _maximum(local_peaks[oi], other), fx.Float32(1e-10)
                            )
                            / 448.0
                        )
                        if (row_lane == 0) & (wave_n == 0):
                            bo.buffer_store(
                                scale,
                                aux,
                                rows[oi] * (intermediate // 128) + nbase // 128,
                            )
                        for ni in range_constexpr(4):
                            value = z[(mi * 4 + ni) * 4 + i]
                            q = fx.Int32(
                                rocdl.cvt_pk_fp8_f32(
                                    fx.Int32.ir_type,
                                    value / scale,
                                    fx.Float32(0.0),
                                    fx.Int32(0),
                                    False,
                                )
                            )
                            bo.buffer_store(
                                fx.Int8(q),
                                out,
                                rows[oi] * intermediate
                                + nbase
                                + wave_n * 64
                                + ni * 16
                                + row_lane,
                            )
                if workers:
                    fx.gpu.barrier()  # Protect shared peaks before the next persistent task.
            else:
                parts = []
                for part in range_constexpr(output_tile // slice_n):
                    g, u = gemm(part * slice_n)
                    parts.append(g / (1.0 + fx.exp2(-g * LOG2_E)) * u)
                for mi in range_constexpr(num_m):
                    for i in range_constexpr(4):
                        oi = mi * 4 + i
                        (values, peak) = ([], fx.Float32(0.0))
                        for ni in range_constexpr(8):
                            index = (mi * num_n + ni % num_n) * 4 + i
                            value = parts[ni // num_n][index]
                            value = fx.Float32(
                                fx.arith.select(
                                    tokens[oi] < received, value, fx.Float32(0.0)
                                )
                            )
                            values.append(value)
                            peak = _maximum(peak, _maximum(value, -value))
                        for shift in range_constexpr(4):
                            peak = _maximum(
                                peak,
                                peak.shuffle_xor(fx.Int32(1 << shift), fx.Int32(16)),
                            )
                        scale = _maximum(peak, fx.Float32(1e-10)) / 448.0
                        if row_lane == 0:
                            bo.buffer_store(
                                scale,
                                aux,
                                rows[oi] * (intermediate // 128) + nbase // 128,
                            )
                        for ni in range_constexpr(8):
                            q = fx.Int32(
                                rocdl.cvt_pk_fp8_f32(
                                    fx.Int32.ir_type,
                                    values[ni] / scale,
                                    fx.Float32(0.0),
                                    fx.Int32(0),
                                    False,
                                )
                            )
                            bo.buffer_store(
                                fx.Int8(q),
                                out,
                                rows[oi] * intermediate + nbase + ni * 16 + row_lane,
                            )

        def dispatch_task(task):
            tile = task // ntiles
            descriptor = _i32(experts, tile, uniform=True)
            expert = _expert_id(descriptor, reuse_metadata)
            if const_expr(cache_policy == WeightCache.ADAPTIVE):

                def has_reuse():
                    if const_expr(reuse_metadata):
                        return _expert_reuse(descriptor)
                    previous = _i32(
                        experts, (tile > 0).select(tile - 1, fx.Int32(0)), uniform=True
                    )
                    following = _i32(experts, tile + 1, uniform=True)
                    return ((tile > 0) & (previous == expert)) | (
                        (tile + 1 < padded // block_m) & (following == expert)
                    )

                if has_reuse():
                    compute_task(task, expert, True)
                else:
                    compute_task(task, expert, False)
            else:
                compute_task(task, expert, cache_policy)

        if workers:
            for task in range(
                fx.Int32(fx.block_idx.x), padded // block_m * ntiles, workers
            ):
                dispatch_task(task)
        else:
            task = fx.Int32(fx.block_idx.x)
            if task < padded // block_m * ntiles:
                dispatch_task(task)

    @fc.jit
    def launch(
        A: fx.Tensor,
        W: fx.Tensor,
        SA: fx.Tensor,
        SW: fx.Tensor,
        IDS: fx.Tensor,
        EXPERTS: fx.Tensor,
        VALID: fx.Tensor,
        COUNT: fx.Tensor,
        AUX: fx.Tensor,
        OUT: fx.Tensor,
        stream: fx.Stream,
    ):
        capacity = fx.Int32(fx.get_scalar(fx.size(A.shape))) // hidden
        local_experts = fx.Int32(fx.get_scalar(fx.size(W.shape))) // (
            2 * intermediate * hidden
        )
        max_tiles = (
            capacity * topk + local_experts * (block_m - 1) + block_m - 1
        ) // block_m
        grid_x = workers if workers else max_tiles * (intermediate // 128)
        kernel(A, W, SA, SW, IDS, EXPERTS, VALID, COUNT, AUX, OUT).launch(
            grid=(grid_x, 1, 1), block=(64 * waves, 1, 1), stream=stream
        )

    return launch
