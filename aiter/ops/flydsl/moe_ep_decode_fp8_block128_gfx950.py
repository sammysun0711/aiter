# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public host entry for rank-local FP8 Block128 MoE on gfx950.

Execution: sort -> gate_up -> optional silu_quant -> down -> optional local_reduce.
The kernels package separates shared primitives, routing and the GEMM stages.
This module owns input checks, branch selection, workspaces and compilation plans.
"""
import functools
import torch
import aiter
import flydsl.compiler as fc
from .kernels.moe_ep_decode_fp8_block128_gfx950.sort import local_sort_launcher
from .kernels.moe_ep_decode_fp8_block128_gfx950.stage1_gate_up import (
    gate_up_fused_launcher,
    gate_up_projection_launcher,
    silu_quant_launcher,
)
from .kernels.moe_ep_decode_fp8_block128_gfx950.stage2_down import down_launcher
from .kernels.moe_ep_decode_fp8_block128_gfx950.reduce import local_reduce_launcher

BLOCK_M = 32
STAGE2_N = 64
STAGE2_GRID_MULT = 8


def _use_local_sort(capacity, global_experts, local_experts, topk):
    return capacity <= 1024 and global_experts <= 1024 and local_experts <= 64 and topk <= 8


def _flat(tensor):
    if tensor.dtype == torch.float8_e4m3fn:
        tensor = tensor.view(torch.int8)
    return tensor.view(-1)


class _Plan:
    def __init__(
        self,
        hidden,
        intermediate,
        topk,
        compute_units,
        block_m,
        projection,
        local_capacity,
        capacity,
        global_experts,
        local_experts,
        device,
    ):
        self.sort = (
            local_sort_launcher(
                capacity,
                topk,
                global_experts,
                local_experts,
                block_m,
                hidden if block_m == 16 else 0,
                device,
            )
            if _use_local_sort(capacity, global_experts, local_experts, topk)
            else None
        )
        if projection:
            self.gate_up = gate_up_projection_launcher(
                hidden, intermediate, block_m, local_capacity, 1, 8 * compute_units, 64
            )
            self.silu_quant = silu_quant_launcher(
                intermediate, local_capacity, 1, 4 * compute_units
            )
        else:
            self.gate_up = gate_up_fused_launcher(hidden, intermediate, topk, block_m)
        self.down = down_launcher(
            hidden, intermediate, topk, STAGE2_GRID_MULT * compute_units, block_m, STAGE2_N
        )
        self.local_reduce = local_reduce_launcher(hidden, topk)
        self.compiled = {}

    def call(self, name, launcher, args):
        # Physical buffer layouts are static. Pointers, receive counts and stream
        # remain runtime values. compile() invokes the launcher once itself.
        if name not in self.compiled:
            values = [fc.from_torch_tensor(t) if isinstance(t, torch.Tensor) else t for t in args]
            options = {} if name == "sort" else {"fast_fp_math": True}
            self.compiled[name] = fc.compile[options](launcher, *values)
        else:
            self.compiled[name](*args)


@functools.lru_cache(None)
def _plan(
    hidden,
    intermediate,
    topk,
    local_experts,
    global_experts,
    capacity,
    device,
    block_m,
    projection,
    local_capacity,
):
    properties = torch.cuda.get_device_properties(device)
    if not properties.gcnArchName.startswith("gfx950"):
        raise ValueError("This FP8 MFMA kernel currently requires gfx950")
    aiter.logger.info(
        "FlyDSL EP decode FP8 Block128 MoE (gfx950): H=%s I=%s local_experts=%s topk=%s capacity=%s M=%s N2=%s projection=%s down=transpose atomic=%s sort=%s",
        hidden,
        intermediate,
        local_experts,
        topk,
        capacity,
        block_m,
        STAGE2_N,
        projection,
        block_m == 16,
        "local" if _use_local_sort(capacity, global_experts, local_experts, topk) else "opus",
    )
    return _Plan(
        hidden,
        intermediate,
        topk,
        properties.multi_processor_count,
        block_m,
        projection,
        local_capacity,
        capacity,
        global_experts,
        local_experts,
        device,
    )


@functools.lru_cache(None)
def _is_gfx950(device):
    return getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").startswith("gfx950")


def _validate_inputs(
    hidden_states,
    w1,
    w2,
    topk_ids,
    topk_weights,
    input_scale,
    w1_scale,
    w2_scale,
    expert_mask,
    num_recv_tokens,
    num_tokens_before_dispatch=None,
):
    tensors = (
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weights,
        input_scale,
        w1_scale,
        w2_scale,
        expert_mask,
        num_recv_tokens,
    )
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("All inputs, scales, mask and receive count must be tensors")
    if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or topk_ids.ndim != 2:
        raise ValueError("Expected matrix inputs/IDs and three-dimensional expert weights")
    if num_tokens_before_dispatch is not None and (
        not isinstance(num_tokens_before_dispatch, int) or num_tokens_before_dispatch < 0
    ):
        raise ValueError("The optional workload hint must be a nonnegative CPU integer")
    capacity, hidden = hidden_states.shape
    local_experts, output_hidden, intermediate = w2.shape
    topk = topk_ids.shape[1]
    if not 1 <= topk <= 32 or not hidden_states.is_cuda:
        raise ValueError("Expected CUDA inputs and top-k in [1,32]")
    if not _is_gfx950(hidden_states.device):
        raise ValueError("This FP8 MFMA kernel currently requires gfx950")
    if capacity <= 0 or local_experts <= 0 or hidden <= 0 or intermediate <= 0:
        raise ValueError("Physical capacity and model dimensions must be positive")
    if hidden != output_hidden or hidden % 1024 or intermediate % 128:
        raise ValueError("Expected hidden multiple of1024 and intermediate multiple of128")
    if w1.shape != (local_experts, 2 * intermediate, hidden):
        raise ValueError("Invalid gate/up weight shape")
    if any(not t.is_contiguous() or t.device != hidden_states.device for t in tensors):
        raise ValueError("Inputs must be contiguous and on the same GPU")
    if any(t.numel() * t.element_size() >= 2**31 for t in tensors):
        raise ValueError("This decode kernel requires each input buffer below2GiB")
    if (
        hidden_states.dtype != torch.float8_e4m3fn
        or w1.dtype != hidden_states.dtype
        or w2.dtype != hidden_states.dtype
    ):
        raise ValueError("Expected FP8 e4m3fn activations and weights")
    if (
        topk_ids.dtype != torch.int32
        or expert_mask.dtype != torch.int32
        or num_recv_tokens.dtype != torch.int32
    ):
        raise ValueError("IDs, expert mask and receive count must be int32")
    if any(t.dtype != torch.float32 for t in (topk_weights, input_scale, w1_scale, w2_scale)):
        raise ValueError("Routing weights and scales must be FP32")
    if (
        topk_ids.shape != (capacity, topk)
        or topk_weights.shape != topk_ids.shape
        or num_recv_tokens.numel() != 1
    ):
        raise ValueError("Invalid routing/count shape")
    if input_scale.shape != (capacity, hidden // 128):
        raise ValueError("Invalid input scale shape")
    if w1_scale.shape != (
        local_experts,
        2 * intermediate // 128,
        hidden // 128,
    ) or w2_scale.shape != (local_experts, hidden // 128, intermediate // 128):
        raise ValueError("Invalid weight scale shape")

    # Uniform-routing mean over ALL experts, including inactive ones. For P
    # ranks each supplying m rows and owning L disjoint experts, the mean is
    # E[n_e] = (P*m) * topk / (P*L) = m*topk/L. With replicated source routes,
    # X ~ Binomial(m, topk/(P*L)) and n_e = P*X: the mean is unchanged, while
    # variance and tails differ from independent routing.
    # This CPU shape hint only selects M16/M32 and fused/projection paths
    # without GPU synchronization. It may include graph padding and is neither
    # an active-expert mean nor an upper bound on any expert's token count.
    # GPU counts/sorted routes determine actual work; allocation bounds below
    # use physical capacity, top-k and padding, not this mean.
    expected_m = (
        num_tokens_before_dispatch * topk / local_experts
        if num_tokens_before_dispatch is not None
        else None
    )
    block_m = 16 if expected_m is not None and expected_m <= 8 else BLOCK_M
    projection = expected_m is not None and expected_m <= 4
    device = hidden_states.device
    global_experts = expert_mask.numel()
    if expert_mask.ndim != 1 or global_experts < local_experts:
        raise ValueError("Expected a one-dimensional global expert mask")
    sorted_capacity = capacity * topk + global_experts * block_m - topk
    local_capacity = (
        (capacity * topk + local_experts * (block_m - 1) + block_m - 1) // block_m
    ) * block_m
    if sorted_capacity * 4 >= 2**31 or local_capacity * intermediate >= 2**31:
        raise ValueError("Sorting and intermediate workspaces must be below2GiB")
    if capacity * hidden * 2 >= 2**31:
        raise ValueError("The output workspace must be below2GiB")
    if block_m != 16 and capacity * topk * hidden * 2 >= 2**31:
        raise ValueError("The partial-output workspace must be below2GiB")
    if projection and local_capacity * (2 * intermediate) * 4 >= 2**31:
        raise ValueError("The projection workspace must be below2GiB")
    return block_m, projection, global_experts, sorted_capacity, local_capacity


def is_supported_ep_decode_fp8_block128_gfx950_input(
    hidden_states,
    w1,
    w2,
    topk_ids,
    topk_weights,
    input_scale,
    w1_scale,
    w2_scale,
    expert_mask,
    num_recv_tokens,
    num_tokens_before_dispatch=None,
):
    """Metadata-only predicate; never read receive counts or masks onto CPU."""
    try:
        _validate_inputs(
            hidden_states,
            w1,
            w2,
            topk_ids,
            topk_weights,
            input_scale,
            w1_scale,
            w2_scale,
            expert_mask,
            num_recv_tokens,
            num_tokens_before_dispatch,
        )
    except ValueError:
        return False
    return True


def fused_moe_ep_decode_fp8_block128_gfx950(
    hidden_states,
    w1,
    w2,
    topk_ids,
    topk_weights,
    input_scale,
    w1_scale,
    w2_scale,
    expert_mask,
    num_recv_tokens,
    num_tokens_before_dispatch=None,
):
    """Compute rank-local routed MoE; return BF16 [capacity, hidden].

    w1/w2 are AITER-shuffled FP8. Weight scales are [E,N/128,K/128];
    input scales are row-major [capacity,hidden/128]. num_recv_tokens is a
    GPU int32 scalar. Nonlocal top-k scratch slots are never read.
    num_tokens_before_dispatch is an optional CPU shape hint, not a receive count.
    Only the valid output prefix is written.
    Call once before CUDA Graph capture to populate the compilation cache.
    """
    block_m, projection, global_experts, sorted_capacity, local_capacity = _validate_inputs(
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weights,
        input_scale,
        w1_scale,
        w2_scale,
        expert_mask,
        num_recv_tokens,
        num_tokens_before_dispatch,
    )
    capacity, hidden = hidden_states.shape
    local_experts, _, intermediate = w2.shape
    topk = topk_ids.shape[1]
    device = hidden_states.device
    atomic_output = block_m == 16
    plan = _plan(
        hidden,
        intermediate,
        topk,
        local_experts,
        global_experts,
        capacity,
        device.index,
        block_m,
        projection,
        local_capacity,
    )
    stream = torch.cuda.current_stream(device)
    ids = torch.empty(sorted_capacity, device=device, dtype=torch.int32)
    weights = torch.empty(sorted_capacity, device=device, dtype=torch.float32)
    experts = torch.empty(
        (sorted_capacity + block_m - 1) // block_m, device=device, dtype=torch.int32
    )
    valid = torch.empty(2, device=device, dtype=torch.int32)
    output = torch.empty((capacity, hidden), device=device, dtype=torch.bfloat16)
    # Both sorters clear the valid output prefix for BF16 atomic accumulation.
    sort_output = (
        output if atomic_output else torch.empty((0, 0), device=device, dtype=torch.bfloat16)
    )
    if plan.sort is not None:
        sorting = [
            _flat(t)
            for t in (
                topk_ids,
                topk_weights,
                expert_mask,
                num_recv_tokens,
                ids,
                weights,
                experts,
                valid,
                sort_output,
            )
        ] + [stream]
        plan.call("sort", plan.sort, sorting)
    else:
        workspace_bytes = aiter.moe_sorting_opus_get_workspace_size(
            capacity, global_experts, topk, 0
        )
        workspace = (
            torch.empty(workspace_bytes, device=device, dtype=torch.uint8)
            if workspace_bytes
            else None
        )
        aiter.moe_sorting_opus_fwd(
            topk_ids,
            topk_weights,
            ids,
            weights,
            experts,
            valid,
            sort_output,
            global_experts,
            block_m,
            expert_mask,
            num_recv_tokens,
            workspace,
            0,
            None,
        )

    # At most capacity*topk assignments plus31 padding rows per local expert.
    q = torch.empty((local_capacity, intermediate), device=device, dtype=torch.float8_e4m3fn)
    qscale = torch.empty((local_capacity, intermediate // 128), device=device, dtype=torch.float32)
    partial = (
        output
        if atomic_output
        else torch.empty((capacity, topk, hidden), device=device, dtype=torch.bfloat16)
    )
    gate_up_args = [
        _flat(t)
        for t in (
            hidden_states,
            w1,
            input_scale,
            w1_scale,
            ids,
            experts,
            valid,
            num_recv_tokens,
            qscale,
            q,
        )
    ] + [stream]
    down_args = [
        _flat(t)
        for t in (
            q,
            w2,
            qscale,
            w2_scale,
            ids,
            experts,
            valid,
            num_recv_tokens,
            weights,
            partial,
        )
    ] + [stream]
    reduction_args = [
        _flat(t) for t in (partial, topk_ids, expert_mask, topk_weights, num_recv_tokens, output)
    ] + [
        capacity,
        stream,
    ]
    if projection:
        projected = torch.empty(
            (local_capacity, 2 * intermediate), device=device, dtype=torch.float32
        )
        gate_up_args = [
            _flat(t)
            for t in (
                hidden_states,
                w1,
                input_scale,
                w1_scale,
                ids,
                experts,
                valid,
                num_recv_tokens,
                projected,
            )
        ] + [stream]
        silu_quant_args = [
            _flat(t) for t in (projected, ids, valid, num_recv_tokens, q, qscale)
        ] + [stream]
        plan.call("gate_up", plan.gate_up, gate_up_args)
        plan.call("silu_quant", plan.silu_quant, silu_quant_args)
    else:
        plan.call("gate_up", plan.gate_up, gate_up_args)
    plan.call("down", plan.down, down_args)
    if not atomic_output:
        plan.call("local_reduce", plan.local_reduce, reduction_args)
    return output
