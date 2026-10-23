# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 Block128 MoE correctness and performance tests.

One GPU simulates rank ep-1 with capacity C=-m and optional valid rows R<C.
Set AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950=1 for the public FlyDSL path.
Offline tuning: gemm_moe_tune.py --ep-fp8-flydsl -i shapes.csv -o tuned.csv.
"""

import argparse
import os
from unittest.mock import patch

import torch
import aiter
from aiter import dtypes
from aiter.fused_moe import moe_sorting, fused_moe
from aiter.test_common import run_perftest
from aiter.utility.moe_blockscale_benchmark import (
    make_topk,
    torch_moe_blockscale,
    make_blockscale_case,
    check_blockscale_output,
    benchmark_ep_moe,
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
            topk_ids,
            topk_weights,
            E,
            model_dim,
            output_dtype,
            expert_mask=expert_mask,
            num_local_tokens=num_local_tokens,
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
    received_tokens=None,
    hot_experts=None,
):
    if ep < 1 or E % ep:
        raise ValueError(f"Expert count {E} must be divisible by positive EP size {ep}")
    received = token if received_tokens is None else received_tokens
    if not 0 < received <= token or (ep == 1 and received != token):
        raise ValueError("Expected 0 < received <= capacity; unused tails require EP")
    use_flydsl = (
        os.environ.get("AITER_FLYDSL_MOE_EP_DECODE_FP8_BLOCK128_GFX950", "0") == "1"
    )
    if use_flydsl and ep == 1:
        raise ValueError("The FlyDSL EP benchmark requires -ep > 1")
    if ep > 1 and shared_E:
        raise ValueError("The EP blockscale fixture uses routed experts only")

    case = make_blockscale_case(
        dtype,
        token,
        model_dim,
        inter_dim,
        E,
        topk,
        ep=ep,
        received_tokens=received,
        routing=routing,
        hot_experts=hot_experts,
        use_g1u1=use_g1u1,
        shared_E=shared_E,
        scale_blks=scale_blks,
    )
    local_E = E // ep
    inputs, out_ref = case.inputs, case.reference
    w1_scale, w2_scale, a1_scale = case.w1_scale, case.w2_scale, case.a1_scale
    expert_mask, num_local_tokens = case.expert_mask, case.num_local_tokens
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
            assert (
                selected.called
            ), "Requested shape did not select the FlyDSL EP kernel"
    else:
        backend = "asm"
        func = asm_moe_test
        kwargs = dict(
            fc1_scale=w1_scale,
            fc2_scale=w2_scale,
            a1_scale=a1_scale[:received] if ep > 1 else a1_scale.t().contiguous(),
            scale_blk=scale_blks,
            output_dtype=dtype,
            expert_mask=expert_mask,
            num_local_tokens=num_local_tokens,
            a1_scale_is_transposed=ep == 1,
        )
        warmup_out = func(*inputs, **kwargs)

    check_blockscale_output(out_ref, warmup_out[:received], f"{backend} eager", ep)
    if ep > 1:
        output, us = benchmark_ep_moe(func, inputs, kwargs)
        metric = "cuda_graph_event_mean"
    else:
        output, us = run_perftest(func, *inputs, **kwargs)
        metric = "profiler_kernel_mean"
    msg = (
        f"[perf] a8w8 {backend}: {us:>8.2f} us ...... M={token}, "
        f"model_dim={model_dim}, inter_dim={inter_dim}, E={E}, local_E={local_E}, "
        f"ep={ep}, ep_id={ep - 1}, topk={topk}, R={received}, C={token}, "
        f"source_hint={tokens_before_dispatch}, routing={routing}, hot_experts={hot_experts}, dtype={dtype}, metric={metric}"
    )
    check_blockscale_output(out_ref, output[:received], msg, ep)


def main():
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
        help="Optional source-rank hint. Used for profile lookup and untuned tile selection; not derived from -m.",
    )
    parser.add_argument(
        "--routing",
        choices=("randn", "uniform", "hot"),
        default="randn",
        help="Router logits: randn (default), Uniform(5,10) with sigmoid weights, or local hot experts.",
    )

    parser.add_argument(
        "--hot-experts",
        type=int,
        default=None,
        help="Hot routing: number of boosted local experts (default: min(topk, local experts)).",
    )
    parser.add_argument(
        "--received-tokens",
        type=int,
        default=None,
        help="Valid EP input rows R <= -m capacity. Default: R=C.",
    )
    args = parser.parse_args()
    for dtype in args.dtype:
        for m in args.m:
            test_fmoe(
                dtype,
                m,
                args.dim,
                args.idim,
                (128, 128),
                args.expert,
                args.topk,
                use_g1u1=True,
                ep=args.expert_parallelism,
                tokens_before_dispatch=args.tokens_before_dispatch,
                routing=args.routing,
                received_tokens=args.received_tokens,
                hot_experts=args.hot_experts,
            )


if __name__ == "__main__":
    main()
