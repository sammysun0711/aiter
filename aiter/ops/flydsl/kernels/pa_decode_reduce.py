# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Combine normalized PA decode partition outputs without changing their ABI."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .kernels_common import LOG2E
from .pa_decode_common import (
    copy_load as _copy_load,
)
from .pa_decode_common import (
    copy_store as _copy_store,
)
from .pa_decode_common import (
    exp2_f32_fast,
    rcp_f32,
    udiv_const,
    urem_const,
)
from .tensor_shim import ptr_buf_tensor

WARP_SIZE = 64
_FLAT_BUFFER_ELEMENTS = 1 << 30


@functools.lru_cache(maxsize=256)
def compile_pa_decode_sw_reduce(
    *,
    max_context_partition_num: int,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    output_dtype_str: str,
    logits_dtype_str: str = "bf16",
):
    # Partition partials (`logits`) are read at this dtype; defaults to bf16
    # (this kernel's original, still-default behavior for existing callers).
    # Callers with an fp8 query should still keep this at bf16 (fp8 has too
    # little precision for a re-accumulated intermediate) -- only real f16/f32
    # queries should pick a matching non-bf16 value here.
    if logits_dtype_str == "f32":
        LOGITS_DTYPE = fx.Float32
    elif logits_dtype_str == "f16":
        LOGITS_DTYPE = fx.Float16
    else:
        LOGITS_DTYPE = fx.BFloat16
    if output_dtype_str == "f32":
        OUTPUT_DTYPE = fx.Float32
    elif output_dtype_str == "f16":
        OUTPUT_DTYPE = fx.Float16
    else:
        OUTPUT_DTYPE = fx.BFloat16
    block_threads = head_size
    assert block_threads > 0, "head_size must be positive"
    assert block_threads <= 1024, "head_size must fit in one workgroup"
    reduce_width = (
        1
        if max_context_partition_num <= 1
        else 1 << ((max_context_partition_num - 1).bit_length())
    )
    reduce_shuffle_offsets = [off for off in [32, 16, 8, 4, 2, 1] if off < reduce_width]
    red_slots = max(1, (block_threads + WARP_SIZE - 1) // WARP_SIZE)

    @fx.struct
    class SharedStorage:
        red: fx.Array[fx.Float32, red_slots, 16]
        part_weights: fx.Array[fx.Float32, max_context_partition_num, 16]

    @flyc.kernel(known_block_size=(block_threads, 1, 1))
    def pa_decode_sw_reduce_kernel(
        # Raw-pointer kernargs: bare i64 data_ptr() (strides are explicit args).
        output_ptr: fx.Int64,
        exp_sums_ptr: fx.Int64,
        max_logits_ptr: fx.Int64,
        logits_ptr: fx.Int64,
        stride_output_bs: fx.Int32,
        stride_output_len: fx.Int32,
        stride_output_kv_head: fx.Int32,
        stride_output_group_size: fx.Int32,
        stride_exp_sums_seq: fx.Int32,
        stride_exp_sums_head: fx.Int32,
        stride_exp_sums_part: fx.Int32,
        stride_logits_seq: fx.Int32,
        stride_logits_head: fx.Int32,
        stride_logits_part: fx.Int32,
        stride_logits_group: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        batch_idx = fx.Int32(gpu.block_id("x"))
        kv_head_idx = fx.Int32(gpu.block_id("y"))
        eqgs_idx = fx.Int32(gpu.block_id("z"))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red_scratch = lds.red.view(fx.make_layout(red_slots, 1))
        if const_expr(max_context_partition_num > WARP_SIZE):
            part_weights_lds = lds.part_weights.view(
                fx.make_layout(max_context_partition_num, 1)
            )

        def _divide_addr(addr, dtype):
            return fx.logical_divide(
                ptr_buf_tensor(addr, elem=dtype, n=_FLAT_BUFFER_ELEMENTS),
                fx.make_layout(1, 1),
            )

        output = _divide_addr(output_ptr, OUTPUT_DTYPE)
        exp_sums = _divide_addr(exp_sums_ptr, fx.Float32)
        max_logits = _divide_addr(max_logits_ptr, fx.Float32)
        logits = _divide_addr(logits_ptr, LOGITS_DTYPE)

        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        copy_logits = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy32b()
                if LOGITS_DTYPE.width == 32
                else fx.rocdl.BufferCopy16b()
            ),
            LOGITS_DTYPE,
        )
        copy_output = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy32b()
                if OUTPUT_DTYPE.width == 32
                else fx.rocdl.BufferCopy16b()
            ),
            OUTPUT_DTYPE,
        )
        f32_register = fx.make_rmem_tensor(1, fx.Float32)
        logits_register = fx.make_rmem_tensor(1, LOGITS_DTYPE)
        output_register = fx.make_rmem_tensor(1, OUTPUT_DTYPE)

        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_log2e = fx.Float32(LOG2E)
        fm_fast = arith.FastMathFlags.fast

        c_w = fx.Int32(WARP_SIZE)
        c_wave_mask = fx.Int32(WARP_SIZE - 1)
        c_red_slots = fx.Int32(red_slots)
        lane = tid & c_wave_mask
        wave = fx.Int32(tid >> fx.Int32(6))

        def _wave_reduce_max_full(val):
            red = val
            for sh in [32, 16, 8, 4, 2, 1]:
                red = fx.max(red, red.shuffle_xor(fx.Int32(sh), c_w))
            return red

        def _wave_reduce_sum_full(val):
            red = val
            for sh in [32, 16, 8, 4, 2, 1]:
                red = red.addf(
                    red.shuffle_xor(fx.Int32(sh), c_w),
                    fastmath=fm_fast,
                )
            return red

        def _block_reduce(val, mode):
            if const_expr(red_slots == 1):
                return (
                    _wave_reduce_max_full(val)
                    if const_expr(mode == "max")
                    else _wave_reduce_sum_full(val)
                )

            neutral = c_neg_inf if const_expr(mode == "max") else c_zero_f
            w = (
                _wave_reduce_max_full(val)
                if const_expr(mode == "max")
                else _wave_reduce_sum_full(val)
            )

            if lane == 0:
                wave_idx = fx.Int32(wave)
                fx.memref_store(w, red_scratch, wave_idx)
            gpu.barrier()

            if wave == 0:
                in_range = lane < c_red_slots
                lane_safe = in_range.select(lane, 0)
                lane_safe_idx = fx.Int32(lane_safe)
                red_val = fx.memref_load(red_scratch, lane_safe_idx)
                red_val = in_range.select(red_val, neutral)
                red_val = (
                    _wave_reduce_max_full(red_val)
                    if const_expr(mode == "max")
                    else _wave_reduce_sum_full(red_val)
                )
                if lane == 0:
                    fx.memref_store(red_val, red_scratch, 0)
            gpu.barrier()

            return fx.memref_load(red_scratch, 0)

        if const_expr(max_context_partition_num <= WARP_SIZE):
            c_part_num = fx.Int32(max_context_partition_num)
            c_reduce_width = fx.Int32(reduce_width)

            def _wave_reduce_max(val):
                red = val
                for sh in reduce_shuffle_offsets:
                    red = fx.max(red, red.shuffle_xor(fx.Int32(sh), c_w))
                return red

            def _wave_reduce_sum(val):
                red = val
                for sh in reduce_shuffle_offsets:
                    red = red.addf(
                        red.shuffle_xor(fx.Int32(sh), c_w),
                        fastmath=fm_fast,
                    )
                return red

            lane_in_range = lane < c_part_num
            lane_in_reduce = lane < c_reduce_width
            part_sum = c_zero_f
            part_max = c_neg_inf
            if lane_in_reduce:
                part_i32 = lane_in_range.select(lane, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = _copy_load(exp_sums, es_off, copy_f32, f32_register)[0]
                part_max_raw = _copy_load(max_logits, es_off, copy_f32, f32_register)[0]
                part_sum = lane_in_range.select(part_sum_raw, c_zero_f)
                part_max = lane_in_range.select(part_max_raw, c_neg_inf)

            global_max = _wave_reduce_max(part_max)
            part_scale = lane_in_range.select(
                exp2_f32_fast((part_max - global_max) * c_log2e), c_zero_f
            )
            scaled_sum = part_sum * part_scale
            global_exp_sum = _wave_reduce_sum(scaled_sum)
            safe_global_exp_sum = (global_exp_sum > c_zero_f).select(
                global_exp_sum, c_one_f
            )
            inv_global_exp_sum = rcp_f32(safe_global_exp_sum)
            weight_local = scaled_sum * inv_global_exp_sum
            weight_local_i32 = weight_local.bitcast(fx.Int32)

            acc = c_zero_f
            for part_idx in range_constexpr(max_context_partition_num):
                part_i32 = fx.Int32(part_idx)
                bcast_addr = part_i32 * 4
                weight_i32 = rocdl.ds_bpermute(
                    T.i32, bcast_addr.ir_value(), weight_local_i32.ir_value()
                )
                weight = fx.Int32(weight_i32).bitcast(fx.Float32)
                logits_off = (
                    batch_idx * stride_logits_seq
                    + kv_head_idx * stride_logits_head
                    + part_i32 * stride_logits_part
                    + eqgs_idx * stride_logits_group
                    + tid
                )
                part_logits_raw = _copy_load(
                    logits, logits_off, copy_logits, logits_register
                )[0]
                part_logits = fx.Float32(part_logits_raw)
                acc = acc + part_logits * weight
        else:
            # More than one wave is needed when NP exceeds 64.
            global_max = c_neg_inf
            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = fx.Int32(chunk_size)
                c_chunk_base = fx.Int32(chunk_base)
                in_chunk = tid < c_chunk_size
                part_i32 = in_chunk.select(tid + c_chunk_base, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_max_raw = _copy_load(max_logits, es_off, copy_f32, f32_register)[0]
                part_max = in_chunk.select(part_max_raw, c_neg_inf)
                chunk_max = _block_reduce(part_max, "max")
                global_max = fx.max(global_max, chunk_max)

            global_exp_sum = c_zero_f
            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = fx.Int32(chunk_size)
                c_chunk_base = fx.Int32(chunk_base)
                in_chunk = tid < c_chunk_size
                part_i32 = in_chunk.select(tid + c_chunk_base, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = _copy_load(exp_sums, es_off, copy_f32, f32_register)[0]
                part_max_raw = _copy_load(max_logits, es_off, copy_f32, f32_register)[0]
                part_sum = in_chunk.select(part_sum_raw, c_zero_f)
                part_max = in_chunk.select(part_max_raw, c_neg_inf)
                part_scale = in_chunk.select(
                    exp2_f32_fast((part_max - global_max) * c_log2e), c_zero_f
                )
                chunk_sum = _block_reduce(part_sum * part_scale, "sum")
                global_exp_sum = global_exp_sum + chunk_sum

            safe_global_exp_sum = (global_exp_sum > c_zero_f).select(
                global_exp_sum, c_one_f
            )
            inv_global_exp_sum = rcp_f32(safe_global_exp_sum)

            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = fx.Int32(chunk_size)
                c_chunk_base = fx.Int32(chunk_base)
                in_chunk = tid < c_chunk_size
                part_i32 = in_chunk.select(tid + c_chunk_base, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = _copy_load(exp_sums, es_off, copy_f32, f32_register)[0]
                part_max_raw = _copy_load(max_logits, es_off, copy_f32, f32_register)[0]
                part_sum = in_chunk.select(part_sum_raw, c_zero_f)
                part_max = in_chunk.select(part_max_raw, global_max)
                part_scale = exp2_f32_fast((part_max - global_max) * c_log2e)
                weight = part_sum * part_scale * inv_global_exp_sum
                if in_chunk:
                    part_idx_idx = fx.Int32(part_i32)
                    fx.memref_store(weight, part_weights_lds, part_idx_idx)

            gpu.barrier()

            acc = c_zero_f
            for part_idx in range_constexpr(max_context_partition_num):
                part_i32 = fx.Int32(part_idx)
                part_idx_idx = fx.Int32(part_idx)
                weight = fx.memref_load(part_weights_lds, part_idx_idx)
                logits_off = (
                    batch_idx * stride_logits_seq
                    + kv_head_idx * stride_logits_head
                    + part_i32 * stride_logits_part
                    + eqgs_idx * stride_logits_group
                    + tid
                )
                part_logits_raw = _copy_load(
                    logits, logits_off, copy_logits, logits_register
                )[0]
                part_logits = fx.Float32(part_logits_raw)
                acc = acc + part_logits * weight

        query_idx = udiv_const(eqgs_idx, query_group_size)
        group_idx = urem_const(eqgs_idx, query_group_size)
        out_off = (
            batch_idx * stride_output_bs
            + query_idx * stride_output_len
            + kv_head_idx * stride_output_kv_head
            + group_idx * stride_output_group_size
            + tid
        )
        out_val = acc if const_expr(output_dtype_str == "f32") else acc.to(OUTPUT_DTYPE)
        _copy_store(
            output,
            out_off,
            copy_output,
            output_register,
            fx.Vector.from_elements([out_val], dtype=OUTPUT_DTYPE),
        )

    @flyc.jit
    def launch_pa_decode_sw_reduce(
        output: fx.Int64,
        exp_sums: fx.Int64,
        max_logits: fx.Int64,
        logits: fx.Int64,
        stride_output_bs,
        stride_output_len,
        stride_output_kv_head,
        stride_output_group_size,
        stride_exp_sums_seq,
        stride_exp_sums_head,
        stride_exp_sums_part,
        stride_logits_seq,
        stride_logits_head,
        stride_logits_part,
        stride_logits_group,
        batch_size,
        num_kv_heads,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL ABI default.
    ):
        pa_decode_sw_reduce_kernel(
            output,
            exp_sums,
            max_logits,
            logits,
            stride_output_bs,
            stride_output_len,
            stride_output_kv_head,
            stride_output_group_size,
            stride_exp_sums_seq,
            stride_exp_sums_head,
            stride_exp_sums_part,
            stride_logits_seq,
            stride_logits_head,
            stride_logits_part,
            stride_logits_group,
        ).launch(
            grid=(batch_size, num_kv_heads, query_seq_len * query_group_size),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return {
        "launch": launch_pa_decode_sw_reduce,
        "kernel": pa_decode_sw_reduce_kernel,
    }
