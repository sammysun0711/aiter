# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Local-expert counting sort for small block-FP8 EP decode inputs.

One CTA builds a global-to-local expert map, counts local assignments, pads
each expert to a GEMM tile and scatters packed (slot, token) IDs. Other CTAs
optionally clear the output for BF16 atomic accumulation. The two operations
write disjoint buffers; completion of this kernel precedes the GEMMs.
"""
import functools

import flydsl.compiler as fc
import flydsl.expr as fx
from flydsl.expr import buffer_ops as bo, range_constexpr, rocdl
from flydsl.expr.utils.arith import _to_raw
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

from .common import (
    _buffer,
    _i32,
    _pack_route,
    EXPERT_REUSE,
    PADDED_COUNT,
    RECEIVED_COUNT,
    ROUTE_MASK_OFFSET,
)


def _wave_prefix(value, lane, steps=6):
    for shift in range_constexpr(steps):
        source = ((lane - (1 << shift)) & 63) * 4
        other = fx.Int32(
            rocdl.ds_bpermute(fx.Int32.ir_type, source.ir_value(), value.ir_value())
        )
        value = value + (lane >= (1 << shift)).select(other, fx.Int32(0))
    return value


def _binary_prefix(value):
    """Inclusive prefix of a binary expert mask without six lane shuffles."""
    bits = fx.Int64(rocdl.ballot(fx.Int64.ir_type, (value != 0).ir_value()))
    low, high = fx.Int32(bits), fx.Int32(bits >> 32)
    prefix = fx.Int32(
        rocdl.mbcnt_lo(fx.Int32.ir_type, low.ir_value(), fx.Int32(0).ir_value())
    )
    prefix = fx.Int32(
        rocdl.mbcnt_hi(fx.Int32.ir_type, high.ir_value(), prefix.ir_value())
    )
    return prefix + value


def _add_one(pointer):
    raw = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.ptrtoint(pointer).ir_value())
    return fx.Int32(
        llvm.atomicrmw(
            llvm.AtomicBinOp.add,
            raw,
            fx.Int32(1).ir_value(),
            llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
        )
    )


@functools.lru_cache(None)
def local_sort_launcher(
    capacity,
    topk,
    global_experts,
    local_experts,
    block_m,
    clear_hidden,
    device,
    reuse_metadata=False,
    route_masks=False,
):
    """GPU receive count bounds work; capacity is only the physical buffer size.

    The binary mask must have exactly local_experts entries set, in weight order.
    Repeated expert IDs are separate assignments. Zero-weight routes are omitted.
    The host chooses this specialization for capacity <= 2048 and top-k <= 8.
    """
    assert 0 < capacity <= 2048 and 0 < topk <= 8
    assert 0 < local_experts <= 64 and local_experts <= global_experts <= 1024
    assert block_m in (16, 32, 64) and clear_hidden % 8 == 0
    assert not route_masks or topk in (1, 2, 4, 8)

    @fx.struct
    class Storage:
        mapping: fx.Array[fx.Int32, global_experts, 16]
        counts: fx.Array[fx.Int32, 64, 16]
        cursors: fx.Array[fx.Int32, 64, 16]
        starts: fx.Array[fx.Int32, 65, 16]
        scratch: fx.Array[fx.Int32, 8, 16]

    @fc.kernel(
        name=f"flydsl_moe_ep_decode_fp8_b128_gfx950_sort_t512_dev{device}",
        known_block_size=[512, 1, 1],
    )
    def kernel(
        IDS: fx.Tensor,
        W: fx.Tensor,
        MASK: fx.Tensor,
        COUNT: fx.Tensor,
        SORTED: fx.Tensor,
        SORTW: fx.Tensor,
        EXPERTS: fx.Tensor,
        VALID: fx.Tensor,
        OUT: fx.Tensor,
    ):
        thread = fx.Int32(fx.thread_idx.x)
        block = fx.Int32(fx.block_idx.x)
        if clear_hidden > 0:
            received_for_clear = _i32(_buffer(COUNT), 0, uniform=True)
            for clear_base in range(
                fx.Int32(0), received_for_clear * clear_hidden // 8, fx.Int32(256 * 512)
            ):
                clear_index = clear_base + block * 512 + thread
                bo.buffer_store(
                    fx.full(4, 0, fx.Int32),
                    _buffer(OUT),
                    clear_index * 4,
                    mask=clear_index < received_for_clear * clear_hidden // 8,
                )

        if block == 0:
            lane, wave = thread % 64, thread // 64
            lds = fx.SharedAllocator().allocate(Storage).peek()
            mapping, counts, starts = lds.mapping.ptr, lds.counts.ptr, lds.starts.ptr
            cursors, scratch = lds.cursors.ptr, lds.scratch.ptr
            received = _i32(_buffer(COUNT), 0, uniform=True)
            if thread < 64:
                fx.ptr_store(fx.Int32(0), counts + fx.Int64(thread))

            # Prefix of the binary mask maps arbitrary global IDs to the order
            # of the local expert weights; no host mask readback is needed.
            carry = fx.Int32(0)
            for chunk in range_constexpr((global_experts + 511) // 512):
                index = chunk * 512 + thread
                active = _i32(
                    _buffer(MASK),
                    (index < global_experts).select(index, fx.Int32(global_experts)),
                )
                prefix = _binary_prefix(active)
                if lane == 63:
                    fx.ptr_store(prefix, scratch + fx.Int64(wave))
                fx.gpu.barrier()
                preceding, total = fx.Int32(0), fx.Int32(0)
                for w in range_constexpr(8):
                    value = fx.ptr_load(scratch + fx.Int64(w))
                    preceding = preceding + (wave > w).select(value, fx.Int32(0))
                    total = total + value
                if index < global_experts:
                    local = (active != 0).select(
                        carry + preceding + prefix - 1, fx.Int32(-1)
                    )
                    fx.ptr_store(local, mapping + fx.Int64(index))
                carry = carry + total
                fx.gpu.barrier()

            def local_route(offset):
                valid = offset < received * topk
                safe = valid.select(offset, fx.Int32(capacity * topk))
                expert = _i32(_buffer(IDS), safe)
                weight = _i32(_buffer(W), safe).bitcast(fx.Float32)
                in_range = (expert >= 0) & (expert < global_experts)
                local = fx.ptr_load(
                    mapping + fx.Int64(in_range.select(expert, fx.Int32(0)))
                )
                selected = valid & in_range & (local >= 0) & (weight != 0.0)
                return local, weight, selected

            # Shared counters cover local experts only, regardless of capacity.
            for base in range(fx.Int32(0), received * topk, fx.Int32(512)):
                local, weight, selected = local_route(base + thread)
                if route_masks:
                    # A power-of-two top-k row stays within one wave. Keep its
                    # valid local slots for the final reduction, including zeros.
                    bits = fx.Int64(rocdl.ballot(fx.Int64.ir_type, _to_raw(selected)))
                    if (thread % topk == 0) & (base + thread < received * topk):
                        row_mask = fx.Int32(
                            (bits >> fx.Int64(lane)) & ((1 << topk) - 1)
                        )
                        bo.buffer_store(
                            row_mask,
                            _buffer(VALID),
                            ROUTE_MASK_OFFSET + (base + thread) // topk,
                        )
                if selected:
                    _add_one(counts + fx.Int64(local))
            fx.gpu.barrier()
            if thread < local_experts:
                actual = fx.ptr_load(counts + fx.Int64(thread))
                padded = ((actual + block_m - 1) // block_m) * block_m
                prefix = _wave_prefix(padded, lane, (local_experts - 1).bit_length())
                fx.ptr_store(prefix - padded, starts + fx.Int64(thread))
                # Keep the histogram intact; only scatter cursors are mutable.
                fx.ptr_store(prefix - padded, cursors + fx.Int64(thread))
                if thread == local_experts - 1:
                    fx.ptr_store(prefix, starts + fx.Int64(local_experts))
                    bo.buffer_store(prefix, _buffer(VALID), PADDED_COUNT)
                    bo.buffer_store(received, _buffer(VALID), RECEIVED_COUNT)
            fx.gpu.barrier()

            # Four lanes per expert initialize tile IDs and the padding tail.
            if thread < local_experts * 4:
                expert, part = thread // 4, thread % 4
                start = fx.ptr_load(starts + fx.Int64(expert))
                end = fx.ptr_load(starts + fx.Int64(expert + 1))
                actual = fx.ptr_load(counts + fx.Int64(expert))
                for tile_i in range(
                    start // block_m + part, end // block_m, fx.Int32(4)
                ):
                    if reuse_metadata:
                        # Low 16 bits: expert ID. Bit 16: enough rows to reuse
                        # weights beyond one tile (at least half an M16 tail).
                        # Plain OPUS descriptors contain only the expert ID.
                        bo.buffer_store(
                            expert
                            | (actual >= block_m + 8).select(
                                fx.Int32(EXPERT_REUSE), fx.Int32(0)
                            ),
                            _buffer(EXPERTS),
                            tile_i,
                        )
                    else:
                        bo.buffer_store(expert, _buffer(EXPERTS), tile_i)
                for pad_i in range(start + actual + part, end, fx.Int32(4)):
                    bo.buffer_store(fx.Int32(capacity), _buffer(SORTED), pad_i)
                    bo.buffer_store(fx.Float32(0), _buffer(SORTW), pad_i)

            for base in range(fx.Int32(0), received * topk, fx.Int32(512)):
                offset = base + thread
                local, weight, selected = local_route(offset)
                if selected:
                    position = _add_one(cursors + fx.Int64(local))
                    bo.buffer_store(
                        _pack_route(offset // topk, offset % topk),
                        _buffer(SORTED),
                        position,
                    )
                    bo.buffer_store(weight, _buffer(SORTW), position)

    @fc.jit
    def launch(
        IDS: fx.Tensor,
        W: fx.Tensor,
        MASK: fx.Tensor,
        COUNT: fx.Tensor,
        SORTED: fx.Tensor,
        SORTW: fx.Tensor,
        EXPERTS: fx.Tensor,
        VALID: fx.Tensor,
        OUT: fx.Tensor,
        stream: fx.Stream,
    ):
        kernel(IDS, W, MASK, COUNT, SORTED, SORTW, EXPERTS, VALID, OUT).launch(
            grid=(256 if clear_hidden else 1, 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    return launch
