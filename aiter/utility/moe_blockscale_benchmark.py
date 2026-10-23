# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared FP8 Block128 fixtures, reference and graph timing for offline tools.

Importing this module does not change the default device or run a benchmark.
Inference does not import it. Tensor generation follows test_moe_blockscale.
"""

import copy
import math
from dataclasses import dataclass
from statistics import mean

import torch
import torch.nn.functional as F
from einops import rearrange
import aiter
from aiter import dtypes, pertoken_quant
from aiter.fused_moe import fused_topk
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose


def make_topk(hidden_states, experts, topk, ep, routing, hot_experts=None):
    if routing not in ("randn", "uniform", "hot"):
        raise ValueError(f"Unknown routing: {routing}")
    if hot_experts is not None and (
        routing != "hot"
        or type(hot_experts) is not int
        or not 1 <= hot_experts <= experts // ep
    ):
        raise ValueError(
            "hot_experts requires hot routing and 1 <= count <= local experts"
        )
    shape = (hidden_states.shape[0], experts)
    if routing == "uniform":
        # MiMo fake dispatch: FP32 Uniform(5,10) logits, distinct top-k IDs,
        # and normalized sigmoid weights. This is not exact load balancing.
        scores = torch.empty(
            shape, dtype=torch.float32, device=hidden_states.device
        ).uniform_(5, 10)
    else:
        scores = torch.randn(
            shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        if routing == "hot":
            local_experts = experts // ep
            start = (ep - 1) * local_experts
            count = min(topk, local_experts) if hot_experts is None else hot_experts
            scores[:, start : start + count] += 20
    weights, ids = fused_topk(hidden_states, scores, topk, True)
    if routing == "uniform":
        weights = F.logsigmoid(scores.gather(1, ids.long())).softmax(dim=-1)
    return weights, ids


def torch_moe_blockscale(
    hidden_states,
    w1,  # [expert, inter_dim*2, model_dim]
    w2,  # [expert, model_dim, inter_dim]
    topk_weight,
    topk_ids,
    dtype,
    # following for quant
    scale_blks=(128, 128),
    a_scale=None,
    # [expert, inter_dim/blk_m, model_dim/blk_k]
    fc1_scale=None,
    # [expert, model_dim/blk_m, inter_dim/blk_k]
    fc2_scale=None,
    expert_mask=None,
):
    computeType = dtypes.fp32
    hidden_states = hidden_states.to(computeType)
    w1 = w1.to(computeType)
    w2 = w2.to(computeType)
    token_num, topk = topk_ids.shape
    expert, model_dim, inter_dim = w2.shape
    B, D = hidden_states.shape
    topk = topk_weight.shape[1]
    if expert_mask is not None:
        local_expert_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
        local_expert_hash[expert_mask == 0] = -1
        topk_ids = local_expert_hash[topk_ids]

    blk_n, blk_k = scale_blks
    if a_scale is not None:
        # print(f'{a_scale.unsqueeze(-1).shape=}, {hidden_states.view(token_num, -1, blk_k).shape=}')
        hidden_states = hidden_states.view(token_num, -1, blk_k) * a_scale.unsqueeze(-1)
        hidden_states = hidden_states.view(token_num, -1)

    hidden_states = hidden_states.view(token_num, 1, model_dim).repeat(1, topk, 1)
    out = torch.zeros(
        (B, topk, D),
        dtype=computeType,
        device=hidden_states.device,
    )
    if w2.shape[2] * 2 == w1.shape[1]:
        moeType = "g1u1"
    else:
        moeType = "g1u0"

    nblk_n = inter_dim // blk_n
    nblk_k = model_dim // blk_k
    if fc1_scale is not None:
        # gose to quant D_w8a8/w8a8
        # blk_n, blk_k = scale_blks
        # expert, nblk_n, nblk_k = fc1_scale.shape
        fc1_scale = rearrange(
            fc1_scale.view(-1, 1)
            .repeat(1, blk_n * blk_k)
            .view(expert, -1, nblk_k, blk_n, blk_k),
            "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
        )
        fc2_scale = rearrange(
            fc2_scale.view(-1, 1)
            .repeat(1, blk_n * blk_k)
            .view(expert, nblk_k, nblk_n, blk_k, blk_n),
            "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
        )
        w1 = w1 * fc1_scale
        w2 = w2 * fc2_scale

    for E_id in range(w1.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = hidden_states[mask]
            act_input = sub_tokens @ (w1[E_id].transpose(0, 1))
            if moeType == "g1u1":
                gate, up = act_input.split([inter_dim, inter_dim], dim=-1)
                act_out = F.silu(gate) * up
            else:
                act_out = F.gelu(act_input)
            out[mask] = act_out @ (w2[E_id].transpose(0, 1))

    return (out * topk_weight.view(B, -1, 1)).sum(dim=1).to(dtype)


def check_blockscale_output(reference, output, msg, ep):
    assert torch.isfinite(output).all(), f"{msg}: non-finite output"
    if ep == 1:
        checkAllclose(reference, output, rtol=0.05, atol=0.05, msg=msg)
        return
    # This reference keeps the intermediate activation in FP32; both FP8
    # backends quantize it. Use the full-shape MoE validation's NSD criteria.
    ref, actual = reference.float(), output.float()
    diff = (ref - actual).square()
    energy = ref.square() + actual.square()
    nsd = (diff.sum() / energy.sum().clamp_min(1e-20)).item()
    row_nsd = (diff.sum(-1) / energy.sum(-1).clamp_min(1e-20)).max().item()
    assert nsd < 1e-3 and row_nsd < 2e-3, (msg, nsd, row_nsd)
    aiter.logger.info(f"{msg} passed: global_nsd={nsd:.6g}, max_row_nsd={row_nsd:.6g}")
    return dict(global_nsd=nsd, max_row_nsd=row_nsd)


@dataclass
class BlockscaleCase:
    inputs: tuple
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    a1_scale: torch.Tensor
    expert_mask: torch.Tensor | None
    num_local_tokens: torch.Tensor | None
    reference: torch.Tensor
    received: int
    ep: int

    def flydsl_args(self):
        x, w1, w2, weights, ids = self.inputs
        experts, hidden, intermediate = w2.shape
        return (
            x,
            w1,
            w2,
            ids,
            weights,
            self.a1_scale,
            self.w1_scale.view(experts, 2 * intermediate // 128, hidden // 128),
            self.w2_scale.view(experts, hidden // 128, intermediate // 128),
            self.expert_mask,
            self.num_local_tokens,
        )

    def public_kwargs(self, source_tokens=None):
        args = self.flydsl_args()
        return dict(
            w1_scale=args[6],
            w2_scale=args[7],
            a1_scale=self.a1_scale,
            expert_mask=self.expert_mask,
            num_local_tokens=self.num_local_tokens,
            num_tokens_before_dispatch=source_tokens,
            quant_type=aiter.QuantType.per_1x128,
            activation=aiter.ActivationType.Silu,
            dtype=self.reference.dtype,
            ep_has_fake_route=False,
        )


def make_blockscale_case(
    dtype,
    token,
    model_dim,
    inter_dim,
    E,
    topk,
    *,
    ep=1,
    received_tokens=None,
    routing="randn",
    hot_experts=None,
    device="cuda",
    use_g1u1=True,
    shared_E=0,
    scale_blks=(128, 128),
):
    """Create one last-rank EP fixture; C=token and R=received_tokens.

    R and routing affect the fixture, not the runtime profile key. Source-token
    hints are passed separately to the kernel and never inferred from capacity.
    """
    if ep < 1 or E % ep:
        raise ValueError("Global experts must be divisible by positive EP size")
    received = token if received_tokens is None else received_tokens
    if not 0 < received <= token or (ep == 1 and received != token):
        raise ValueError("Expected 0 < received <= capacity; unused tails require EP")
    if ep > 1 and shared_E:
        raise ValueError("The EP fixture uses routed experts only")
    with torch.device(device):
        local_E = E // ep
        expert_mask = None
        num_local_tokens = None
        if ep > 1:
            torch.manual_seed(0)
            expert_mask = torch.zeros(E, dtype=dtypes.i32, device="cuda")
            expert_mask[(ep - 1) * local_E : ep * local_E] = 1
            num_local_tokens = torch.tensor([received], dtype=dtypes.i32, device="cuda")

        input = torch.randn((token, model_dim), dtype=dtype)
        if use_g1u1:
            w1 = (
                torch.randn((local_E + shared_E, inter_dim * 2, model_dim), dtype=dtype)
                / 10
            )
        else:
            w1 = torch.randn((local_E + shared_E, inter_dim, model_dim), dtype=dtype)
        w2 = torch.randn((local_E + shared_E, model_dim, inter_dim), dtype=dtype) / 10
        topk_weights, topk_ids = make_topk(input, E, topk, ep, routing, hot_experts)
        # Report actual local work outside timing; hot routing changes both expert
        # concentration and the number of local assignments for a fixed M.
        counts = torch.bincount(topk_ids[:received].flatten().long(), minlength=E)
        counts = counts[(ep - 1) * local_E : ep * local_E]
        aiter.logger.info(
            f"[routing] mode={routing}, active_local_experts={(counts > 0).sum().item()}, "
            f"local_assignments={counts.sum().item()}, max_tokens_per_expert={counts.max().item()}"
        )

        scale_blk_n, scale_blk_k = scale_blks
        quant_dtype = dtypes.fp8

        # block quant w1
        tmp = rearrange(
            w1.view(
                -1,
                w1.shape[1] // scale_blk_n,
                scale_blk_n,
                w1.shape[2] // scale_blk_k,
                scale_blk_k,
            ),
            "e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)",
        ).contiguous()
        w1_q, w1_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
        w1_q = rearrange(
            w1_q.view(
                -1,
                w1.shape[1] // scale_blk_n,
                w1.shape[2] // scale_blk_k,
                scale_blk_n,
                scale_blk_k,
            ),
            "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
        ).contiguous()
        w1_scale = w1_scale.view(local_E + shared_E, -1)

        # block quant w2
        tmp = rearrange(
            w2.view(
                -1,
                model_dim // scale_blk_n,
                scale_blk_n,
                inter_dim // scale_blk_k,
                scale_blk_k,
            ),
            "e num_blk_n blk_n num_blk_k blk_k -> e num_blk_n num_blk_k (blk_n blk_k)",
        ).contiguous()
        w2_q, w2_scale = pertoken_quant(tmp, quant_dtype=quant_dtype)
        w2_q = rearrange(
            w2_q.view(
                -1,
                w2.shape[1] // scale_blk_n,
                w2.shape[2] // scale_blk_k,
                scale_blk_n,
                scale_blk_k,
            ),
            "e num_blk_n num_blk_k blk_n blk_k -> e (num_blk_n blk_n) (num_blk_k blk_k)",
        ).contiguous()
        w2_scale = w2_scale.view(local_E + shared_E, -1)

        # block quant input
        a1_q, a1_scale = pertoken_quant(
            input.view(-1, model_dim // scale_blk_k, scale_blk_k),
            quant_dtype=quant_dtype,
        )
        a1_q = a1_q.view(-1, model_dim)
        a1_scale = a1_scale.squeeze(-1)

        reference = torch_moe_blockscale(
            a1_q[:received],
            w1_q,
            w2_q,
            topk_weights[:received],
            topk_ids[:received],
            dtype,
            scale_blks=scale_blks,
            fc1_scale=w1_scale,
            fc2_scale=w2_scale,
            a_scale=a1_scale[:received],
            expert_mask=expert_mask,
        )
        w1 = shuffle_weight(w1_q, (16, 16))
        w2 = shuffle_weight(w2_q, (16, 16))
        w1.is_shuffled = w2.is_shuffled = True
        return BlockscaleCase(
            (a1_q, w1, w2, topk_weights, topk_ids),
            w1_scale,
            w2_scale,
            a1_scale,
            expert_mask,
            num_local_tokens,
            reference,
            received,
            ep,
        )


def benchmark_ep_moe(func, inputs, kwargs, *, warmup=5, iters=101):
    """Mean GPU-event time for a complete local MoE, in microseconds.

    A captured graph contains ``iters`` rounds of four distinct input/weight
    address sets. Five graph replays are averaged. ``warmup`` counts untimed
    rounds before capture; reference checks stay outside the timed graph.
    """
    if warmup < 0 or iters < 1:
        raise ValueError("warmup must be nonnegative and iters must be positive")
    rotated = [copy.deepcopy((inputs, kwargs)) for _ in range(3)] + [(inputs, kwargs)]
    torch.cuda.synchronize()
    result = [None]

    def run():
        for args, options in rotated:
            result[0] = func(*args, **options)

    with torch.cuda.stream(torch.cuda.Stream(device=inputs[0].device)):
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iters):
                run()
        timings = []
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end) * 1000 / (iters * len(rotated)))
    us = mean(timings)
    if not math.isfinite(us) or us <= 0:
        raise RuntimeError("GPU event timing returned no measurable duration")
    return result[0], us
