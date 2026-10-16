# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 Block128 MoE correctness and performance tests.

For EP, one GPU simulates rank ``ep - 1`` with R=C=M.
--routing selects randn logits, fake-dispatch uniform logits, or local hot experts.
The default backend is ASM. Set AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950=1
before starting Python to test the FlyDSL backend through fused_moe instead.

Example: python op_tests/test_moe_blockscale.py -ep 8 -e 384 -k 8 \
    -dim 6144 -idim 2048 -m 672 --tokens-before-dispatch 84
"""

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import checkAllclose, run_perftest
from aiter.fused_moe import torch_moe, moe_sorting, fused_topk, fused_moe
from aiter.ops.shuffle import shuffle_weight
from aiter import pertoken_quant
from aiter import dtypes
from einops import rearrange
import argparse
import copy
import os
from unittest.mock import patch

BLOCK_SIZE_M = 32


def make_topk(hidden_states, experts, topk, ep, routing):
    shape = (hidden_states.shape[0], experts)
    if routing == "uniform":
        # MiMo fake dispatch: FP32 Uniform(5,10) logits, distinct top-k IDs,
        # and normalized sigmoid weights. This is not exact load balancing.
        scores = torch.empty(shape, dtype=torch.float32, device=hidden_states.device).uniform_(5, 10)
    else:
        scores = torch.randn(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        if routing == "hot":
            local_experts = experts // ep
            start = (ep - 1) * local_experts
            # Keep the fixed logical rank busy; no extra hotspot parameters.
            scores[:, start : start + min(topk, local_experts)] += 20
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


def torch_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weight,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,  # [expert, inter_dim, 1]
    fc2_scale=None,  # [expert, model_dim, 1]
    fc1_smooth_scale=None,  # [expert, 1, model_dim]
    fc2_smooth_scale=None,  # [expert, 1, inter_dim]
):
    return torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale,
        fc2_scale,
        fc1_smooth_scale,
        fc2_smooth_scale,
    )


def asm_moe_test(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    # following for int8 quant
    fc1_scale=None,
    fc2_scale=None,
    a1_scale=None,
    scale_blk=(128, 128),
    output_dtype=dtypes.bf16,
    expert_mask=None,
    num_local_tokens=None,
    a1_scale_is_transposed=True,
):

    model_dim = hidden_states.shape[-1]
    topk = topk_ids.shape[-1]
    E = expert_mask.numel() if expert_mask is not None else w1.shape[0]
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out_asm = (
        moe_sorting(
            topk_ids, topk_weights, E, model_dim, output_dtype,
            expert_mask=expert_mask, num_local_tokens=num_local_tokens,
        )
    )
    if not a1_scale_is_transposed:
        # EP backends receive the same row-major scales. Include ASM's layout
        # conversion in its measured path, as in fused_moe's blockscale adapter.
        a1_scale = a1_scale.t().contiguous()
    scale_blk_n, scale_blk_k = scale_blk
    aiter.fmoe_fp8_blockscale_g1u1(
        out_asm,
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        topk,
        a1_scale,
        fc1_scale,
        fc2_scale,
        "",
        scale_blk_n,
        scale_blk_k,
        None,
    )
    return out_asm


torch.set_default_device("cuda")


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


def benchmark_ep_moe(func, inputs, kwargs):
    # HIP events include driver-launched FlyDSL kernels that torch.profiler
    # does not consistently report. Use the same four-copy graph for both paths.
    from triton.testing import do_bench_cudagraph

    rotated = [copy.deepcopy((inputs, kwargs)) for _ in range(3)] + [(inputs, kwargs)]
    # do_bench_cudagraph uses its own stream; finish the input copies first.
    torch.cuda.synchronize()
    result = [None]

    def run():
        for args, options in rotated:
            result[0] = func(*args, **options)

    us = do_bench_cudagraph(run, rep=40, return_mode="mean") * 1000 / len(rotated)
    assert us > 0, "GPU event timing returned no measurable duration"
    return result[0], us


def test_fmoe(
    dtype,
    token,
    model_dim,
    inter_dim,
    scale_blks,
    E,
    topk,
    quant="No",
    use_g1u1=False,
    shared_E=0,
    ep=1,
    tokens_before_dispatch=None,
    routing="randn",
):
    if ep < 1 or E % ep:
        raise ValueError(f"Expert count {E} must be divisible by positive EP size {ep}")
    use_flydsl = os.environ.get(
        "AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950", "0"
    ) == "1"
    if use_flydsl and ep == 1:
        raise ValueError("The FlyDSL EP benchmark requires -ep > 1")
    if ep > 1 and shared_E:
        raise ValueError("The EP blockscale fixture uses routed experts only")

    # Follow test_moe_ep.py: one GPU simulates the last logical EP rank.
    # M is the input row count, not a reconstructed global/source batch.
    local_E = E // ep
    expert_mask = None
    num_local_tokens = None
    if ep > 1:
        torch.manual_seed(0)
        expert_mask = torch.zeros(E, dtype=dtypes.i32, device="cuda")
        expert_mask[(ep - 1) * local_E : ep * local_E] = 1
        # First version has no unused receive-buffer tail: R = C = M.
        num_local_tokens = torch.tensor([token], dtype=dtypes.i32, device="cuda")

    input = torch.randn((token, model_dim), dtype=dtype)
    if use_g1u1:
        w1 = torch.randn((local_E + shared_E, inter_dim * 2, model_dim), dtype=dtype) / 10
    else:
        w1 = torch.randn((local_E + shared_E, inter_dim, model_dim), dtype=dtype)
    w2 = torch.randn((local_E + shared_E, model_dim, inter_dim), dtype=dtype) / 10
    topk_weights, topk_ids = make_topk(input, E, topk, ep, routing)
    # Report actual local work outside timing; hot routing changes both expert
    # concentration and the number of local assignments for a fixed M.
    counts = torch.bincount(topk_ids.flatten().long(), minlength=E)
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
        input.view(-1, model_dim // scale_blk_k, scale_blk_k), quant_dtype=quant_dtype
    )
    a1_q = a1_q.view(-1, model_dim)
    a1_scale = a1_scale.squeeze(-1)

    # w2, fc2_scale = pertoken_quant(w2, quant_dtype=quant_dtype)

    out_ref, us_ref = run_perftest(
        torch_moe_blockscale,
        a1_q,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        dtype,
        scale_blks=scale_blks,
        fc1_scale=w1_scale,
        fc2_scale=w2_scale,
        a_scale=a1_scale,
        expert_mask=expert_mask,
        num_warmup=1,
        num_iters=2,
    )
    # out_ref2, us_ref = run_perftest(torch_moe,
    #                                 input,
    #                                 w1,
    #                                 w2,
    #                                 topk_weights,
    #                                 topk_ids,
    #                                 num_warmup=1, num_iters=2)
    # msg = '111'
    # checkAllclose(out_ref, out_ref2, rtol=0.01, atol=100, msg=msg)

    w1_shuffled = shuffle_weight(w1_q, (16, 16))
    w2_shuffled = shuffle_weight(w2_q, (16, 16))
    w1_shuffled.is_shuffled = w2_shuffled.is_shuffled = True
    inputs = (a1_q, w1_shuffled, w2_shuffled, topk_weights, topk_ids)
    if use_flydsl:
        backend = "flydsl"
        func = fused_moe
        kwargs = dict(
            w1_scale=w1_scale.view(local_E, 2 * inter_dim // 128, model_dim // 128),
            w2_scale=w2_scale.view(local_E, model_dim // 128, inter_dim // 128),
            a1_scale=a1_scale,
            expert_mask=expert_mask,
            num_local_tokens=num_local_tokens,
            num_tokens_before_dispatch=tokens_before_dispatch,
            quant_type=aiter.QuantType.per_1x128,
            activation=aiter.ActivationType.Silu,
            dtype=dtype,
            ep_has_fake_route=False,
        )
        from aiter.ops.flydsl import moe_ep_decode_fp8_block128_gfx950 as flydsl_backend

        # Check the public dispatch once outside timing. Never report a native
        # fallback as a FlyDSL result when the requested shape is unsupported.
        with patch.object(
            flydsl_backend,
            "fused_moe_ep_decode_fp8_block128_gfx950",
            wraps=flydsl_backend.fused_moe_ep_decode_fp8_block128_gfx950,
        ) as selected:
            warmup_out = func(*inputs, **kwargs)
            assert selected.called, "Requested shape did not select the FlyDSL EP kernel"
    else:
        backend = "asm"
        func = asm_moe_test
        kwargs = dict(
            fc1_scale=w1_scale,
            fc2_scale=w2_scale,
            a1_scale=a1_scale if ep > 1 else a1_scale.t().contiguous(),
            scale_blk=scale_blks,
            output_dtype=dtype,
            expert_mask=expert_mask,
            num_local_tokens=num_local_tokens,
            a1_scale_is_transposed=ep == 1,
        )
        warmup_out = func(*inputs, **kwargs)

    check_blockscale_output(out_ref, warmup_out, f"{backend} eager", ep)
    if ep > 1:
        output, us = benchmark_ep_moe(func, inputs, kwargs)
        metric = "cuda_graph_event_mean"
    else:
        output, us = run_perftest(func, *inputs, **kwargs)
        metric = "profiler_kernel_mean"
    msg = (
        f"[perf] a8w8 {backend}: {us:>8.2f} us ...... M={token}, "
        f"model_dim={model_dim}, inter_dim={inter_dim}, E={E}, local_E={local_E}, "
        f"ep={ep}, ep_id={ep - 1}, topk={topk}, R={token}, C={token}, "
        f"source_hint={tokens_before_dispatch}, routing={routing}, dtype={dtype}, metric={metric}"
    )
    check_blockscale_output(out_ref, output, msg, ep)


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    default=[dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)

parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[1, 2, 5, 16, 32, 163840],
    help="""M of mnk.
    e.g.: -m 32""",
)

parser.add_argument(
    "-dim",
    type=int,
    default=7168,
    help="""Hidden dimension. Default is 7168.
    e.g.: -dim 7168""",
)

parser.add_argument(
    "-idim",
    type=int,
    default=256,
    help="""Intermediate dimension. Default is 256.
    e.g.: -idim 256""",
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    default=256,
    help="""Number of experts. Default is 256.
    e.g.: -e 256""",
)
parser.add_argument(
    "-k",
    "--topk",
    type=int,
    default=8,
    help="""Top-k value. Default is 8.
    e.g.: -k 8""",
)
parser.add_argument(
    "-ep",
    "--expert-parallelism",
    type=int,
    default=1,
    help="Logical EP size; simulate its last rank on one GPU (default: 1).",
)
parser.add_argument(
    "--tokens-before-dispatch",
    type=int,
    default=None,
    help="Optional source-rank hint (>=4 for FlyDSL); None selects M32. Not derived from -m.",
)
parser.add_argument(
    "--routing",
    choices=("randn", "uniform", "hot"),
    default="randn",
    help="Router logits: randn (default), Uniform(5,10) with sigmoid weights, or local hot experts.",
)

args = parser.parse_args()

for dtype in args.dtype:
    for m in args.m:
        for dim in [args.dim]:
            for idim in [args.idim]:
                scale_blks = (128, 128)
                test_fmoe(
                    dtype,
                    m,
                    dim,
                    idim,
                    scale_blks,
                    args.expert,
                    args.topk,
                    quant="No",
                    use_g1u1=True,
                    ep=args.expert_parallelism,
                    tokens_before_dispatch=args.tokens_before_dispatch,
                    routing=args.routing,
                )
