# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block-FP8 down projection with contiguous four-column MFMA outputs."""
import functools
import flydsl.compiler as fc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from .common import _buffer, _i32, _fragment, _streaming_fragment, _dot, _scale
from .configuration import WeightCache
from .common import PADDED_COUNT
from .common import _route_token, _route_slot, _expert_id, _expert_reuse


def _write_atomic(out, values, token, col, hidden):
    """Accumulate two packed BF16 pairs into an already-cleared output."""
    for pair in range_constexpr(2):
        packed = Vec.from_elements(values[pair * 2 : pair * 2 + 2], fx.BFloat16)
        offset = (token * hidden + col + pair * 2) * 2
        rocdl.raw_ptr_buffer_atomic_fadd(packed, out, offset, fx.Int32(0), fx.Int32(0))


def _write_partial(out, values, token, slot, col, hidden, topk):
    """Each expert owns its original top-k slot; no atomic is needed here."""
    offset = (token * topk + slot) * hidden + col
    bo.buffer_store(Vec.from_elements(values, fx.BFloat16), out, offset)


@functools.lru_cache(None)
def down_launcher(
    hidden,
    intermediate,
    topk,
    workers,
    block_m=32,
    block_n=64,
    cache_policy=WeightCache.STREAM,
    adaptive_m=False,
    reuse_metadata=False,
    atomic_output=False,
):
    reuse_cache = cache_policy == WeightCache.ADAPTIVE
    cached_weights = cache_policy == WeightCache.CACHE
    num_m, num_n = block_m // 16, block_n // 16
    atomic = atomic_output
    suffix = "atomic" if atomic else "partial"

    @fc.kernel(
        name=f"flydsl_moe_ep_decode_fp8_b128_gfx950_s2_m{block_m}n{block_n}k128_transpose_{suffix}_c{int(cached_weights)}_a1_am{int(adaptive_m)}_rc{int(reuse_cache)}",
        known_block_size=[64, 1, 1],
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
        WEIGHTS: fx.Tensor,
        OUT: fx.Tensor,
    ):
        lane = fx.Int32(fx.thread_idx.x)
        row_lane, k_lane = lane % 16, lane // 16
        ar, wr, sar, swr = _buffer(A), _buffer(W), _buffer(SA), _buffer(SW)
        ids, experts, weights, out = (
            _buffer(IDS),
            _buffer(EXPERTS),
            _buffer(WEIGHTS),
            _buffer(OUT),
        )
        padded = _i32(_buffer(VALID), PADDED_COUNT, uniform=True)
        received = _i32(_buffer(COUNT), 0, uniform=True)
        ntiles = hidden // block_n
        for task in range(
            fx.Int32(fx.block_idx.x), (padded // block_m) * ntiles, workers
        ):
            mtile, nbase = task // ntiles, (task % ntiles) * block_n
            descriptor = _i32(experts, mtile, uniform=True)
            expert = _expert_id(descriptor, reuse_metadata)
            rows = [
                mtile * block_m + mi * 16 + row_lane for mi in range_constexpr(num_m)
            ]
            routes = [_i32(ids, row) for row in rows]
            tokens = [_route_token(route) for route in routes]

            def compute_rows(active_m, cache_tile):
                def load_activation(kb):
                    activations, scales = [], []
                    for mi in range_constexpr(active_m):
                        activations.append(
                            _fragment(
                                ar,
                                row_lane * intermediate + k_lane * 16,
                                64,
                                tokens[mi] < received,
                                soffset_bytes=(mtile * block_m + mi * 16) * intermediate
                                + kb * 128,
                            )
                        )
                        scales.append(
                            fx.Float32(
                                bo.buffer_load(
                                    sar,
                                    row_lane * (intermediate // 128),
                                    soffset_bytes=(
                                        (mtile * block_m + mi * 16)
                                        * (intermediate // 128)
                                        + kb
                                    )
                                    * 4,
                                    vec_width=1,
                                    dtype=fx.Float32,
                                    mask=tokens[mi] < received,
                                )
                            )
                        )
                    return (
                        Vec.from_elements(
                            [v[i] for v in activations for i in range_constexpr(8)],
                            fx.Int32,
                        ),
                        Vec.from_elements(scales, fx.Float32),
                    )

                def load(kb):
                    weight_tiles = []
                    sw = _scale(
                        swr,
                        (expert * (hidden // 128) + nbase // 128)
                        * (intermediate // 128)
                        + kb,
                    )
                    av, sa = load_activation(kb)
                    for ni in range_constexpr(num_n):
                        # Keep the wave-uniform tile/K base in scalar registers;
                        # the per-lane packed-weight offset stays fixed across K.
                        scalar_base = expert * hidden * intermediate
                        scalar_base += (
                            nbase // 16 + ni
                        ) * intermediate * 16 + kb * 2048
                        fragment = (
                            _streaming_fragment(
                                wr, row_lane * 16 + k_lane * 256, 1024, scalar_base
                            )
                            if reuse_cache and not cache_tile
                            else _fragment(
                                wr,
                                row_lane * 16 + k_lane * 256,
                                1024,
                                non_temporal=not cache_tile,
                                soffset_bytes=scalar_base,
                            )
                        )
                        weight_tiles.append(fragment)
                    return (
                        av,
                        Vec.from_elements(
                            [v[i] for v in weight_tiles for i in range_constexpr(8)],
                            fx.Int32,
                        ),
                        sa,
                        sw,
                    )

                def step(acc, av, wv, sa, sw):
                    values = []
                    for mi in range_constexpr(active_m):
                        a = Vec.from_elements(
                            [av[mi * 8 + i] for i in range_constexpr(8)], fx.Int32
                        )
                        for ni in range_constexpr(num_n):
                            w = Vec.from_elements(
                                [wv[ni * 8 + i] for i in range_constexpr(8)], fx.Int32
                            )
                            old = Vec.from_elements(
                                [
                                    acc[(mi * num_n + ni) * 4 + i]
                                    for i in range_constexpr(4)
                                ],
                                fx.Float32,
                            )
                            # Exchanging operands gives each lane four adjacent N
                            # outputs for one token, sharing one activation scale.
                            value = old + _dot(w, a) * (sa[mi] * sw)
                            values.extend([value[i] for i in range_constexpr(4)])
                    return Vec.from_elements(values, fx.Float32)

                acc = fx.full(active_m * num_n * 4, 0.0, fx.Float32)
                # Prefetch both weights and activations for the next K128 step.
                av, wv, sa, sw = load(fx.Int32(0))
                for kb in range(intermediate // 128 - 1):
                    an, wn, san, swn = load(kb + 1)

                    acc = step(acc, av, wv, sa, sw)
                    av, wv, sa, sw = an, wn, san, swn

                acc = step(acc, av, wv, sa, sw)
                for mi in range_constexpr(active_m):
                    weight = fx.Float32(
                        bo.buffer_load(weights, rows[mi], vec_width=1, dtype=fx.Float32)
                    )
                    if tokens[mi] < received:
                        for ni in range_constexpr(num_n):
                            values = [
                                fx.BFloat16(acc[(mi * num_n + ni) * 4 + i] * weight)
                                for i in range_constexpr(4)
                            ]
                            col = nbase + ni * 16 + k_lane * 4
                            if const_expr(atomic):
                                # Sorting clears the output. Zero-weight padding
                                # must not atomically accumulate a possible NaN.
                                if weight != 0:
                                    _write_atomic(out, values, tokens[mi], col, hidden)
                            else:
                                _write_partial(
                                    out,
                                    values,
                                    tokens[mi],
                                    _route_slot(routes[mi]),
                                    col,
                                    hidden,
                                    topk,
                                )

            def has_rows(offset):
                return (
                    _route_token(_i32(ids, mtile * block_m + offset, uniform=True))
                    < received
                )

            def compute_tile(cache_tile):
                # Specialize the K loop once for the live M16 fragments. The full
                # tile takes its original dense loop; no per-K padding branches.
                if const_expr(not adaptive_m):
                    compute_rows(num_m, cache_tile)
                elif const_expr(block_m == 16):
                    compute_rows(1, cache_tile)
                elif const_expr(block_m == 32):
                    if has_rows(16):
                        compute_rows(2, cache_tile)
                    else:
                        compute_rows(1, cache_tile)
                else:
                    if has_rows(16):
                        if has_rows(32):
                            if has_rows(48):
                                compute_rows(4, cache_tile)
                            else:
                                compute_rows(3, cache_tile)
                        else:
                            compute_rows(2, cache_tile)
                    else:
                        compute_rows(1, cache_tile)

            if const_expr(reuse_cache):

                def spans_multiple_tiles():
                    if const_expr(reuse_metadata):
                        # Require at least half an MFMA M16 fragment beyond the
                        # first tile; caching a nearly empty tail can lose to
                        # streaming reads despite touching the same weights.
                        return _expert_reuse(descriptor)
                    # OPUS descriptors have no private flags. Its expert groups
                    # remain contiguous, so adjacent IDs provide the same hint.
                    previous = _i32(
                        experts,
                        (mtile > 0).select(mtile - 1, fx.Int32(0)),
                        uniform=True,
                    )
                    next_expert = _i32(experts, mtile + 1, uniform=True)
                    return ((mtile > 0) & (previous == expert)) | (
                        (mtile + 1 < padded // block_m) & (next_expert == expert)
                    )

                repeated = spans_multiple_tiles()
                if repeated:
                    compute_tile(True)
                else:
                    compute_tile(False)
            else:
                compute_tile(cached_weights)

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
        WEIGHTS: fx.Tensor,
        OUT: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(A, W, SA, SW, IDS, EXPERTS, VALID, COUNT, WEIGHTS, OUT).launch(
            grid=(workers, 1, 1), block=(64, 1, 1), stream=stream
        )

    return launch
