#!/usr/bin/env python3
"""Small rocprofiler target for the Qwen3.8 TP8 A16W4 finalists."""

from __future__ import annotations

import argparse

import torch

from aiter.ops.triton.moe.moe_op_gemm_a16w4 import moe_gemm_a16w4
from aiter.ops.triton.moe.moe_routing.routing import routing


BASELINE = {
    "stage1": {
        "block_k": 256,
        "block_m": 128,
        "block_n": 256,
        "group_m": 1,
        "kpack": 1,
        "matrix_instr_nonkdim": 16,
        "num_stages": 1,
        "num_warps": 8,
        "split_k": 1,
        "w_cache_modifier": None,
        "waves_per_eu": 2,
        "xcd_swizzle": 1,
    },
    "stage2": {
        "block_k": 256,
        "block_m": 128,
        "block_n": 512,
        "group_m": 4,
        "kpack": 1,
        "matrix_instr_nonkdim": 16,
        "num_stages": 1,
        "num_warps": 8,
        "split_k": 1,
        "w_cache_modifier": None,
        "waves_per_eu": 0,
        "xcd_swizzle": 8,
    },
}

TUNED = {
    "stage1": {
        "block_k": 128,
        "block_m": 128,
        "block_n": 256,
        "group_m": 1,
        "kpack": 1,
        "matrix_instr_nonkdim": 16,
        "num_stages": 1,
        "num_warps": 8,
        "split_k": 1,
        "w_cache_modifier": None,
        "waves_per_eu": 0,
        "xcd_swizzle": 8,
    },
    "stage2": {
        "block_k": 128,
        "block_m": 128,
        "block_n": 256,
        "group_m": 1,
        "kpack": 1,
        "matrix_instr_nonkdim": 16,
        "num_stages": 2,
        "num_warps": 8,
        "split_k": 1,
        "w_cache_modifier": None,
        "waves_per_eu": 0,
        "xcd_swizzle": 1,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--variant", choices=["baseline", "tuned"], required=True)
    parser.add_argument("--m", type=int, default=8000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    hidden = 8192
    intermediate = 256
    experts = 512
    topk = 10
    torch.manual_seed(8128)

    logits = torch.randn((args.m, experts), device="cuda", dtype=torch.float32)
    routing_data, gather_idx, scatter_idx = routing(logits, topk)
    config = (BASELINE if args.variant == "baseline" else TUNED)[args.stage]

    if args.stage == "stage1":
        x = torch.randn(
            (args.m, hidden), device="cuda", dtype=torch.bfloat16
        )
        weight_storage = torch.randint(
            0,
            256,
            (experts, 2 * intermediate, hidden // 2),
            device="cuda",
            dtype=torch.uint8,
        )
        scale_storage = torch.full(
            (experts, 2 * intermediate, hidden // 32),
            120,
            device="cuda",
            dtype=torch.uint8,
        )
        weight = weight_storage.transpose(1, 2)
        scale = scale_storage.transpose(1, 2)

        def run():
            return moe_gemm_a16w4(
                x,
                weight,
                None,
                scale,
                routing_data=routing_data,
                gather_indx=gather_idx,
                out_dtype=torch.bfloat16,
                kernel_config=config,
            )

    else:
        x = torch.randn(
            (args.m * topk, intermediate), device="cuda", dtype=torch.bfloat16
        )
        weight_storage = torch.randint(
            0,
            256,
            (experts, hidden, intermediate // 2),
            device="cuda",
            dtype=torch.uint8,
        )
        scale_storage = torch.full(
            (experts, hidden, intermediate // 32),
            120,
            device="cuda",
            dtype=torch.uint8,
        )
        weight = weight_storage.transpose(1, 2)
        scale = scale_storage.transpose(1, 2)

        def run():
            return moe_gemm_a16w4(
                x,
                weight,
                None,
                scale,
                routing_data=routing_data,
                scatter_indx=scatter_idx,
                gammas=routing_data.gate_scal,
                out_dtype=torch.bfloat16,
                kernel_config=config,
            )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(f"a16w4-{args.stage}-{args.variant}")
    for _ in range(args.reps):
        output = run()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    print(
        {
            "stage": args.stage,
            "variant": args.variant,
            "m": args.m,
            "expanded_m": int(gather_idx.shape[0]),
            "expert_blocks": int(
                routing_data.n_blocks(gather_idx.shape[0], routing_data.block_m)
            ),
            "config": config,
            "finite": bool(torch.isfinite(output).all().item()),
        }
    )


if __name__ == "__main__":
    main()
