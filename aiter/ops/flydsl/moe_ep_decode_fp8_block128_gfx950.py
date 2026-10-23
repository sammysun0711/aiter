# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public host entry for rank-local FP8 Block128 MoE on gfx950.

Execution: sort -> gate_up -> optional silu_quant -> down -> optional local_reduce.
The kernels package separates shared primitives, routing and the GEMM stages.
This module owns input checks, branch selection, workspaces and compilation plans.
"""
import functools
from dataclasses import dataclass
import torch
import aiter
import flydsl.compiler as fc
from .kernels.moe_ep_decode_fp8_block128_gfx950.configuration import (
    MoeConfig,
    Stage1,
    WeightCache,
    default_config,
)
from .kernels.moe_ep_decode_fp8_block128_gfx950.sort import local_sort_launcher
from .kernels.moe_ep_decode_fp8_block128_gfx950.common import ROUTE_MASK_OFFSET
from .kernels.moe_ep_decode_fp8_block128_gfx950.stage1_fused_fp8_quant import (
    gate_up_fused_fp8_quant_launcher,
)
from .kernels.moe_ep_decode_fp8_block128_gfx950.stage1_separate_fp8_quant import (
    gate_up_gemm_launcher,
    silu_quant_launcher,
)
from .kernels.moe_ep_decode_fp8_block128_gfx950.stage2_down import down_launcher
from .kernels.moe_ep_decode_fp8_block128_gfx950.reduce import local_reduce_launcher


def _use_local_sort(capacity, global_experts, local_experts, topk, source_tokens=None):
    # A padded receive buffer need not imply a large sort. This is only a
    # scheduling hint: the sorter still processes the GPU receive count and
    # supports every count up to the physical capacity.
    workload = (
        capacity
        if source_tokens is None
        else source_tokens * global_experts // local_experts
    )
    # Full-MoE cutoff: local sorting also supplies route masks and exact expert
    # reuse hints to the following kernels. Paired gfx950 EP8/EP16 measurements
    # retain 1024 as a conservative default across routing distributions; the
    # standalone sorter's crossover can occur earlier.
    return (
        capacity <= 2048
        and min(capacity, workload) <= 1024
        and global_experts <= 1024
        and local_experts <= 64
        and topk <= 8
    )


def _flat(tensor):
    if tensor.dtype == torch.float8_e4m3fn:
        tensor = tensor.view(torch.int8)
    return tensor.view(-1)


@dataclass(frozen=True)
class _Shape:
    capacity: int
    hidden: int
    intermediate: int
    topk: int
    local_experts: int
    global_experts: int

    def sorted_capacity(self, block_m):
        return self.capacity * self.topk + self.global_experts * block_m - self.topk

    def local_capacity(self, block_m):
        rows = self.capacity * self.topk + self.local_experts * (block_m - 1)
        return ((rows + block_m - 1) // block_m) * block_m


@dataclass
class SortedRoutes:
    """Named views of the existing sort ABI; no additional GPU allocation.

    counts[0:2] are padded/received rows; optional local slot masks follow.
    expert_tiles are plain OPUS IDs or local IDs with the shared reuse bit.
    """

    token_ids: torch.Tensor
    weights: torch.Tensor
    expert_tiles: torch.Tensor
    counts: torch.Tensor
    output: torch.Tensor
    clear_output: torch.Tensor | None
    workspace: torch.Tensor | None = None

    @property
    def slot_masks(self):
        return self.counts[ROUTE_MASK_OFFSET:]


@dataclass
class _Workspace:
    q: torch.Tensor
    qscale: torch.Tensor
    partial: torch.Tensor
    projected: torch.Tensor | None


class _Plan:
    def __init__(self, shape, config, device, use_local_sort):
        self.shape, self.config = shape, config
        h, inter, k = shape.hidden, shape.intermediate, shape.topk
        m = config.block_m
        cu = torch.cuda.get_device_properties(device).multi_processor_count
        local_capacity = shape.local_capacity(m)
        reuse_metadata = use_local_sort and (
            config.stage2_cache == WeightCache.ADAPTIVE
            or (not config.uses_separate_quant and config.stage1_cache == WeightCache.ADAPTIVE)
        )
        self.route_masks = (
            use_local_sort and not config.uses_atomic_output and k in (1, 2, 4, 8)
        )
        self.sort = (
            local_sort_launcher(
                shape.capacity,
                k,
                shape.global_experts,
                shape.local_experts,
                m,
                h if config.uses_atomic_output else 0,
                device,
                reuse_metadata,
                self.route_masks,
            )
            if use_local_sort
            else None
        )
        if config.uses_separate_quant:
            self.gate_up = gate_up_gemm_launcher(
                h,
                inter,
                m,
                local_capacity,
                1,
                config.stage1_workers * cu,
                config.stage1_n,
                config.stage1_cache,
                config.stage1.transposed,
                reuse_metadata,
            )
            self.silu_quant = silu_quant_launcher(
                inter,
                local_capacity,
                1,
                4 * cu,
                m,
                config.quant_tiled,
            )
        else:
            self.gate_up = gate_up_fused_fp8_quant_launcher(
                h,
                inter,
                k,
                m,
                config.stage1_n,
                config.stage1_cache,
                config.stage1_workers * cu,
                config.stage1 == Stage1.FUSED_FP8_QUANT_PARALLEL,
                reuse_metadata,
            )
        self.down = down_launcher(
            h,
            inter,
            k,
            config.stage2_workers * cu,
            m,
            config.stage2_n,
            cache_policy=config.stage2_cache,
            adaptive_m=config.down_adaptive_m,
            reuse_metadata=reuse_metadata,
            atomic_output=config.uses_atomic_output,
        )
        self.local_reduce = local_reduce_launcher(h, k, self.route_masks)
        self.compiled = {}

    def call(self, name, launcher, tensors, *scalars):
        # All flattening and the current stream are assembled in this one place.
        args = [_flat(t) for t in tensors] + [
            *scalars,
            torch.cuda.current_stream(tensors[0].device),
        ]
        if name not in self.compiled:
            values = [
                fc.from_torch_tensor(t) if isinstance(t, torch.Tensor) else t
                for t in args
            ]
            options = {} if name == "sort" else {"fast_fp_math": True}
            self.compiled[name] = fc.compile[options](launcher, *values)
        else:
            self.compiled[name](*args)


@functools.lru_cache(None)
def _plan(shape, config, device, use_local_sort):
    aiter.logger.info(
        "FlyDSL EP decode FP8 Block128: C=%s H=%s I=%s local_E=%s M=%s stage1=%s output=%s sort=%s",
        shape.capacity,
        shape.hidden,
        shape.intermediate,
        shape.local_experts,
        config.block_m,
        config.stage1.value,
        config.output.value,
        "local" if use_local_sort else "opus",
    )
    return _Plan(shape, config, device, use_local_sort)


@functools.lru_cache(None)
def _is_gfx950(device):
    return getattr(
        torch.cuda.get_device_properties(device), "gcnArchName", ""
    ).startswith("gfx950")


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
    config=None,
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
        raise ValueError(
            "Expected matrix inputs/IDs and three-dimensional expert weights"
        )
    if num_tokens_before_dispatch is not None and (
        not isinstance(num_tokens_before_dispatch, int)
        or num_tokens_before_dispatch < 0
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
        raise ValueError(
            "Expected hidden multiple of1024 and intermediate multiple of128"
        )
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
    if any(
        t.dtype != torch.float32
        for t in (topk_weights, input_scale, w1_scale, w2_scale)
    ):
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

    selected = config or default_config(num_tokens_before_dispatch, topk, local_experts)
    block_m, projection = selected.block_m, selected.uses_separate_quant
    selected.validate()
    device = hidden_states.device
    global_experts = expert_mask.numel()
    # A mask/count pair alone does not imply EP: TP can supply an all-local mask.
    if expert_mask.ndim != 1 or global_experts <= local_experts:
        raise ValueError(
            "Expected an EP expert mask with fewer local than global experts"
        )
    sorted_capacity = capacity * topk + global_experts * block_m - topk
    local_capacity = (
        (capacity * topk + local_experts * (block_m - 1) + block_m - 1) // block_m
    ) * block_m
    if sorted_capacity * 4 >= 2**31 or local_capacity * intermediate >= 2**31:
        raise ValueError("Sorting and intermediate workspaces must be below2GiB")
    if capacity * hidden * 2 >= 2**31:
        raise ValueError("The output workspace must be below2GiB")
    if not selected.uses_atomic_output and capacity * topk * hidden * 2 >= 2**31:
        raise ValueError("The partial-output workspace must be below2GiB")
    if projection and local_capacity * (2 * intermediate) * 4 >= 2**31:
        raise ValueError("The projection workspace must be below2GiB")
    return _Shape(capacity, hidden, intermediate, topk, local_experts, global_experts)


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
    config=None,
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
            config,
        )
    except ValueError:
        return False
    return True


def _sort_routes(plan, ids, weights, mask, count):
    shape, config = plan.shape, plan.config
    device = ids.device
    maximum = shape.sorted_capacity(config.block_m)
    routes = SortedRoutes(
        token_ids=torch.empty(maximum, device=device, dtype=torch.int32),
        weights=torch.empty(maximum, device=device, dtype=torch.float32),
        expert_tiles=torch.empty(
            (maximum + config.block_m - 1) // config.block_m,
            device=device,
            dtype=torch.int32,
        ),
        counts=torch.empty(
            ROUTE_MASK_OFFSET + (shape.capacity if plan.route_masks else 0),
            device=device,
            dtype=torch.int32,
        ),
        output=torch.empty(
            (shape.capacity, shape.hidden), device=device, dtype=torch.bfloat16
        ),
        clear_output=None,
    )
    routes.clear_output = (
        routes.output
        if config.uses_atomic_output
        else torch.empty((0, 0), device=device, dtype=torch.bfloat16)
    )
    if plan.sort is not None:
        plan.call(
            "sort",
            plan.sort,
            (
                ids,
                weights,
                mask,
                count,
                routes.token_ids,
                routes.weights,
                routes.expert_tiles,
                routes.counts,
                routes.clear_output,
            ),
        )
    else:
        size = aiter.moe_sorting_opus_get_workspace_size(
            shape.capacity, shape.global_experts, shape.topk, 0
        )
        routes.workspace = (
            torch.empty(size, device=device, dtype=torch.uint8) if size else None
        )
        aiter.moe_sorting_opus_fwd(
            ids,
            weights,
            routes.token_ids,
            routes.weights,
            routes.expert_tiles,
            routes.counts,
            routes.clear_output,
            shape.global_experts,
            config.block_m,
            mask,
            count,
            routes.workspace,
            0,
            None,
        )
    return routes


def _allocate_workspace(plan, routes):
    shape, config = plan.shape, plan.config
    device = routes.output.device
    local_capacity = shape.local_capacity(config.block_m)
    return _Workspace(
        q=torch.empty(
            (local_capacity, shape.intermediate),
            device=device,
            dtype=torch.float8_e4m3fn,
        ),
        qscale=torch.empty(
            (local_capacity, shape.intermediate // 128),
            device=device,
            dtype=torch.float32,
        ),
        partial=(
            routes.output
            if config.uses_atomic_output
            else torch.empty(
                (shape.capacity, shape.topk, shape.hidden),
                device=device,
                dtype=torch.bfloat16,
            )
        ),
        projected=(
            torch.empty(
                (1, local_capacity, 2 * shape.intermediate),
                device=device,
                dtype=torch.float32,
            )
            if config.uses_separate_quant
            else None
        ),
    )


def _run_stage1(plan, routes, workspace, x, w1, input_scale, w1_scale, count):
    common = (
        x,
        w1,
        input_scale,
        w1_scale,
        routes.token_ids,
        routes.expert_tiles,
        routes.counts,
        count,
    )
    if plan.config.uses_separate_quant:
        plan.call("gate_up", plan.gate_up, (*common, workspace.projected))
        plan.call(
            "silu_quant",
            plan.silu_quant,
            (
                workspace.projected,
                routes.token_ids,
                routes.counts,
                count,
                workspace.q,
                workspace.qscale,
            ),
        )
    else:
        plan.call("gate_up", plan.gate_up, (*common, workspace.qscale, workspace.q))


def _run_down(plan, routes, workspace, w2, w2_scale, count):
    plan.call(
        "down",
        plan.down,
        (
            workspace.q,
            w2,
            workspace.qscale,
            w2_scale,
            routes.token_ids,
            routes.expert_tiles,
            routes.counts,
            count,
            routes.weights,
            workspace.partial,
        ),
    )


def _reduce_output(plan, routes, workspace, ids, mask, weights, count):
    if not plan.config.uses_atomic_output:
        plan.call(
            "local_reduce",
            plan.local_reduce,
            (
                workspace.partial,
                routes.slot_masks if plan.route_masks else ids,
                mask,
                weights,
                count,
                routes.output,
            ),
            plan.shape.capacity,
        )


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
    config=None,
):
    """Compute rank-local routed MoE; return BF16 [capacity, hidden].

    Scales are FP32; weights have AITER's shuffled FP8 layout. Only the GPU
    receive-count prefix is valid. Compile once before graph capture; receive
    counts, masks, routes and tensor contents may then change without CPU reads.
    """
    shape = _validate_inputs(
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
        config,
    )
    config = config or default_config(
        num_tokens_before_dispatch, shape.topk, shape.local_experts
    )
    plan = _plan(
        shape,
        config,
        hidden_states.device.index,
        _use_local_sort(
            shape.capacity,
            shape.global_experts,
            shape.local_experts,
            shape.topk,
            num_tokens_before_dispatch,
        ),
    )
    routes = _sort_routes(plan, topk_ids, topk_weights, expert_mask, num_recv_tokens)
    workspace = _allocate_workspace(plan, routes)
    _run_stage1(
        plan,
        routes,
        workspace,
        hidden_states,
        w1,
        input_scale,
        w1_scale,
        num_recv_tokens,
    )
    _run_down(plan, routes, workspace, w2, w2_scale, num_recv_tokens)
    _reduce_output(
        plan, routes, workspace, topk_ids, expert_mask, topk_weights, num_recv_tokens
    )
    return routes.output
