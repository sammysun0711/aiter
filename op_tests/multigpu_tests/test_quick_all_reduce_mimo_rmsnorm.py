# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import importlib
import multiprocessing as mp
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

import aiter as ops
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port


qr_module = importlib.import_module("aiter.dist.device_communicators.quick_all_reduce")
QuickAllReduce = qr_module.QuickAllReduce
QuickReduceRegime = qr_module.QuickReduceRegime


HIDDEN_DIM = 6144
EPS = 1e-6
FP_QUICK_REDUCE = 0


@pytest.mark.parametrize(
    ("arch", "expected"),
    [("gfx942:sramecc+:xnack-", True), ("gfx950", True), ("gfx90a", False)],
)
def test_quick_reduce_arch_guard(monkeypatch, arch: str, expected: bool):
    props = type("DeviceProperties", (), {"gcnArchName": arch})()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: props)
    assert qr_module.qr_rocm_arch_available() is expected


def test_mimo_rmsnorm_fusion_requires_fp_and_contiguous_input():
    communicator = object.__new__(QuickAllReduce)
    communicator.disabled = True
    communicator.should_quick_allreduce = lambda _: True
    residual = torch.empty((2, HIDDEN_DIM), dtype=torch.bfloat16)
    weight = torch.empty((HIDDEN_DIM,), dtype=torch.bfloat16)

    communicator.qr_quant_level = QuickReduceRegime.FP8
    assert not communicator.should_quick_allreduce_mimo_rmsnorm(
        residual, residual, weight, HIDDEN_DIM
    )

    communicator.qr_quant_level = QuickReduceRegime.FP
    noncontiguous = torch.empty((HIDDEN_DIM, 2), dtype=torch.bfloat16).t()
    assert not noncontiguous.is_contiguous()
    assert not communicator.should_quick_allreduce_mimo_rmsnorm(
        noncontiguous, residual, weight, HIDDEN_DIM
    )


@dataclass(frozen=True)
class Case:
    tokens: int
    dtype: torch.dtype
    cast_bf2half: bool
    warmup: int = 10
    iterations: int = 50


def _init_quick_reduce(rank: int, world_size: int, init_method: str):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    ptr = ops.init_custom_qr(rank, world_size, None)
    handle = ops.qr_get_handle(ptr)
    handles = [None] * world_size
    dist.all_gather_object(handles, handle)
    ops.qr_open_handles(ptr, handles)
    return device, ptr


def _make_inputs(case: Case, rank: int, device: torch.device):
    input_generator = torch.Generator(device="cpu").manual_seed(1000 + rank)
    shared_generator = torch.Generator(device="cpu").manual_seed(2026)
    shape = (case.tokens, HIDDEN_DIM)
    inp = (torch.randn(shape, dtype=case.dtype, generator=input_generator) * 0.1).to(
        device
    )
    residual = (
        torch.randn(shape, dtype=case.dtype, generator=shared_generator) * 0.1
    ).to(device)
    weight = (
        1.0
        + torch.randn((HIDDEN_DIM,), dtype=case.dtype, generator=shared_generator)
        * 0.01
    ).to(device)
    return inp, residual, weight


def _run_unfused(
    ptr,
    inp,
    residual,
    weight,
    cast_bf2half,
    allreduce_out=None,
    residual_out=None,
    out=None,
):
    if allreduce_out is None:
        allreduce_out = torch.empty_like(inp)
    if residual_out is None:
        residual_out = torch.empty_like(residual)
    if out is None:
        out = torch.empty_like(inp)
    ops.qr_all_reduce(
        ptr,
        inp,
        allreduce_out,
        FP_QUICK_REDUCE,
        cast_bf2half,
    )
    ops.rmsnorm2d_fwd_with_add(
        out,
        allreduce_out,
        residual,
        residual_out,
        weight,
        EPS,
    )
    return out, residual_out


def _run_fused(ptr, inp, residual, weight, cast_bf2half, residual_out=None, out=None):
    if residual_out is None:
        residual_out = torch.empty_like(residual)
    if out is None:
        out = torch.empty_like(inp)
    ops.qr_all_reduce_mimo_rmsnorm(
        ptr,
        inp,
        residual,
        residual_out,
        out,
        weight,
        EPS,
        HIDDEN_DIM,
        FP_QUICK_REDUCE,
        cast_bf2half,
    )
    return out, residual_out


def _time_path(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _worker(rank: int, world_size: int, case: Case, init_method: str, benchmark: bool):
    ptr = None
    try:
        device, ptr = _init_quick_reduce(rank, world_size, init_method)
        inp, residual, weight = _make_inputs(case, rank, device)
        dist.barrier()

        ref_out, ref_residual = _run_unfused(
            ptr, inp, residual, weight, case.cast_bf2half
        )
        fused_out, fused_residual = _run_fused(
            ptr, inp, residual, weight, case.cast_bf2half
        )
        torch.cuda.synchronize()

        atol = 6e-3 if case.dtype == torch.float16 else 3e-2
        rtol = atol
        torch.testing.assert_close(
            fused_residual.float(), ref_residual.float(), atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            fused_out.float(), ref_out.float(), atol=atol, rtol=rtol
        )

        result = {
            "rank": rank,
            "out_max_abs_diff": (fused_out.float() - ref_out.float())
            .abs()
            .max()
            .item(),
            "residual_max_abs_diff": (fused_residual.float() - ref_residual.float())
            .abs()
            .max()
            .item(),
        }
        if benchmark:
            unfused_allreduce_out = torch.empty_like(inp)
            unfused_residual_out = torch.empty_like(residual)
            unfused_out = torch.empty_like(inp)
            fused_residual_out = torch.empty_like(residual)
            fused_out = torch.empty_like(inp)
            result["unfused_us"] = _time_path(
                lambda: _run_unfused(
                    ptr,
                    inp,
                    residual,
                    weight,
                    case.cast_bf2half,
                    unfused_allreduce_out,
                    unfused_residual_out,
                    unfused_out,
                ),
                case.warmup,
                case.iterations,
            )
            result["fused_us"] = _time_path(
                lambda: _run_fused(
                    ptr,
                    inp,
                    residual,
                    weight,
                    case.cast_bf2half,
                    fused_residual_out,
                    fused_out,
                ),
                case.warmup,
                case.iterations,
            )
        return result
    finally:
        if ptr is not None:
            ops.qr_destroy(ptr)
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()


def run_case(world_size: int, case: Case, benchmark: bool = False):
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=world_size) as pool:
        pending = [
            pool.apply_async(
                _worker,
                args=(rank, world_size, case, init_method, benchmark),
            )
            for rank in range(world_size)
        ]
        return [item.get(timeout=600) for item in pending]


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires eight GPUs")
@pytest.mark.parametrize(
    "case",
    [
        Case(tokens=1, dtype=torch.bfloat16, cast_bf2half=True),
        Case(tokens=17, dtype=torch.bfloat16, cast_bf2half=True),
        Case(tokens=128, dtype=torch.bfloat16, cast_bf2half=False),
        Case(tokens=257, dtype=torch.float16, cast_bf2half=False),
    ],
)
def test_quick_all_reduce_mimo_rmsnorm_matches_unfused(case: Case):
    run_case(8, case)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=8, choices=(2, 4, 8))
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=(1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--native-bf16", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(
        "tokens,unfused_us,fused_us,speedup_pct,out_max_abs_diff,residual_max_abs_diff"
    )
    for tokens in args.tokens:
        case = Case(
            tokens=tokens,
            dtype=torch.bfloat16,
            cast_bf2half=not args.native_bf16,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        results = run_case(args.world_size, case, benchmark=True)
        unfused_us = max(item["unfused_us"] for item in results)
        fused_us = max(item["fused_us"] for item in results)
        speedup = (unfused_us / fused_us - 1.0) * 100.0
        out_diff = max(item["out_max_abs_diff"] for item in results)
        residual_diff = max(item["residual_max_abs_diff"] for item in results)
        print(
            f"{tokens},{unfused_us:.3f},{fused_us:.3f},{speedup:.2f},"
            f"{out_diff:.6f},{residual_diff:.6f}"
        )
