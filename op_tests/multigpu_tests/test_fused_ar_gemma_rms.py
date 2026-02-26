# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Simple op test for aiter All reduce + Gemma RMSNorm fusion.

Gemma RMSNorm: out = x * rsqrt(mean(x^2) + eps) * (1 + weight)
Reference: test_fused_ar_rms.py (standard RMSNorm fusion).
"""

import os
from typing import Optional
import torch
import torch.distributed as dist
import argparse
import itertools
from aiter import dtypes
from multiprocessing import set_start_method, Pool, freeze_support
import logging

from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    get_tp_group,
    graph_capture,
    destroy_model_parallel,
    destroy_distributed_environment,
)
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_gemma_rmsnorm,
)
from aiter.test_common import (
    checkAllclose,
    perftest,
    benchmark,
)

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def _gemma_rmsnorm_cpu(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma-style RMSNorm: out = x * rsqrt(mean(x^2) + eps) * (1 + weight)."""
    orig_dtype = x.dtype
    x = x.float()
    weight = weight.float()
    rms = (x.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    out = x * rms * (1.0 + weight)
    return out.to(orig_dtype)


def _gemma_rmsnorm_device(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma RMSNorm on device (graph-capture safe)."""
    rms = (x.float().pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    return (x.float() * rms * (1.0 + weight.float())).to(x.dtype)


def fused_ar_gemma_rmsnorm(
    tp_size: int,
    pp_size: int,
    rankID: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    withGraph: bool = False,
    distributed_init_method: Optional[str] = None,
):
    """Run fused allreduce + Gemma RMSNorm on one rank."""
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)
    weight = weight.to(device)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                out, res_out = tensor_model_parallel_fused_allreduce_gemma_rmsnorm(
                    x, x, weight, eps
                )
        out.fill_(0)
        res_out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()
            torch.cuda.synchronize()

        _, us = run_ca()
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()
        return out, us
    else:

        @perftest()
        def run_ca(x):
            out, res_out = tensor_model_parallel_fused_allreduce_gemma_rmsnorm(
                x, x, weight, eps
            )
            return out

        out, us = run_ca(x)
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()
        return out, us


def split_ar_gemma_rmsnorm(
    tp_size: int,
    pp_size: int,
    rankID: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    withGraph: bool = False,
    distributed_init_method: Optional[str] = None,
):
    """Reference: all_reduce then manual Gemma RMSNorm (add + norm)."""
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    logger.info(f"RANK: {rankID} {tp_size} init_process_group (split)...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)
    weight = weight.to(device)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                ar_out = tensor_model_parallel_all_reduce(x)
                residual = ar_out + x
                out = _gemma_rmsnorm_device(residual, weight, eps)
        out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()
            torch.cuda.synchronize()

        _, us = run_ca()
        result = out.clone()
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()
        return result, us
    else:

        @perftest()
        def run_ca(x):
            ar_out = tensor_model_parallel_all_reduce(x)
            residual = ar_out + x
            out = _gemma_rmsnorm_device(residual, weight, eps)
            return out

        out, us = run_ca(x)
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()
        return out, us


@benchmark()
def _run_test_fused_ar_gemma_rmsnorm(
    tp_size: int,
    pp_size: int,
    shape: tuple,
    dtype: torch.dtype,
    withGraph: bool,
    distributed_init_method: Optional[str],
):
    """Compare fused allreduce+Gemma RMSNorm vs CPU reference (Gemma formula)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49374"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    cpu_rslt = []
    weight_list = []
    res_inp = []
    m, n = shape[0], shape[1]
    eps = 1e-6

    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        res_inp.append(x)
        ref += x
        weight = torch.randn((n,), dtype=dtype)
        weight_list.append(weight)
        rets.append(
            pool.apply_async(
                fused_ar_gemma_rmsnorm,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    x,
                    weight,
                    eps,
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()

    # CPU reference: ref = sum(x_i), per-rank normalized = ref + res_inp[i], then Gemma norm
    for i in range(tp_size):
        normalized = ref + res_inp[i]
        host_rslt = _gemma_rmsnorm_cpu(normalized, weight_list[i], eps)
        cpu_rslt.append(host_rslt)

    rets = [el.get() for el in rets]
    for out, us in rets:
        msg = f"test_fused_ar_gemma_rmsnorm: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        checkAllclose(cpu_rslt[out.device.index], out.to(ref), msg=msg)


def test_fused_ar_gemma_rmsnorm(
    tp_size: int = 2,
    pp_size: int = 1,
    shape: tuple = (13, 512),
    dtype: Optional[torch.dtype] = None,
    withGraph: bool = False,
    distributed_init_method: Optional[str] = None,
):
    """Compare fused allreduce+Gemma RMSNorm vs CPU reference (Gemma formula)."""
    if dtype is None:
        dtype = dtypes.d_dtypes["fp16"]
    if distributed_init_method is None:
        from aiter.dist.utils import get_open_port, get_distributed_init_method, get_ip
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    _run_test_fused_ar_gemma_rmsnorm(
        tp_size, pp_size, shape, dtype, withGraph, distributed_init_method
    )


@benchmark()
def _run_test_split_ar_gemma_rmsnorm(
    tp_size: int,
    pp_size: int,
    shape: tuple,
    dtype: torch.dtype,
    withGraph: bool,
    distributed_init_method: Optional[str],
):
    """Compare split (all_reduce + Gemma norm) vs CPU reference."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49375"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    cpu_rslt = []
    weight_list = []
    res_inp = []
    m, n = shape[0], shape[1]
    eps = 1e-6

    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        res_inp.append(x)
        ref += x
        weight = torch.randn((n,), dtype=dtype)
        weight_list.append(weight)
        rets.append(
            pool.apply_async(
                split_ar_gemma_rmsnorm,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    x,
                    weight,
                    eps,
                    withGraph,
                    distributed_init_method,
                ),
            )
        )
    pool.close()
    pool.join()

    for i in range(tp_size):
        normalized = ref + res_inp[i]
        host_rslt = _gemma_rmsnorm_cpu(normalized, weight_list[i], eps)
        cpu_rslt.append(host_rslt)

    rets = [el.get() for el in rets]
    for out, us in rets:
        msg = f"test_split_ar_gemma_rmsnorm: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        checkAllclose(cpu_rslt[out.device.index], out.to(ref), msg=msg)


def test_split_ar_gemma_rmsnorm(
    tp_size: int = 2,
    pp_size: int = 1,
    shape: tuple = (13, 512),
    dtype: Optional[torch.dtype] = None,
    withGraph: bool = False,
    distributed_init_method: Optional[str] = None,
):
    """Compare split (all_reduce + Gemma norm) vs CPU reference."""
    if dtype is None:
        dtype = dtypes.d_dtypes["fp16"]
    if distributed_init_method is None:
        from aiter.dist.utils import get_open_port, get_distributed_init_method, get_ip
        distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
    _run_test_split_ar_gemma_rmsnorm(
        tp_size, pp_size, shape, dtype, withGraph, distributed_init_method
    )


# Fused kernel only supports float32, float16, bfloat16.
# l_shape includes exact GemmaRMSNorm cases: (1, 256), (80, 4096) from forward_native logs.
#l_dtype = ["fp16", "bf16"]
l_dtype = ["bf16"]
#l_shape = [(1, 256), (13, 512), (13, 1024), (13, 2048), (17, 4096), (80, 4096)]
l_shape = [(1, 256), (80, 4096)]
l_tp = [8]
l_pp = [1]
l_graph = [False, True]

parser = argparse.ArgumentParser(description="All reduce + Gemma RMSNorm fusion test")
parser.add_argument(
    "-d", "--dtype", type=str, choices=l_dtype, nargs="?", const=None, default=None
)
parser.add_argument(
    "-s", "--shape", type=dtypes.str2tuple, nargs="?", const=None, default=None
)
parser.add_argument("-t", "--tp", type=int, nargs="?", const=None, default=None)
parser.add_argument("-p", "--pp", type=int, nargs="?", const=None, default=None)
parser.add_argument(
    "-g", "--graphon", type=int, nargs="?", const=None, default=None
)


if __name__ == "__main__":
    freeze_support()
    from aiter.dist.utils import get_open_port, get_distributed_init_method, get_ip

    args = parser.parse_args()
    if args.dtype is None:
        l_dtype_ = [dtypes.d_dtypes[k] for k in l_dtype]
    else:
        l_dtype_ = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        l_shape = [args.shape]
    if args.tp is not None:
        l_tp = [args.tp]
    if args.pp is not None:
        l_pp = [args.pp]
    if args.graphon is not None:
        l_graph = [bool(args.graphon)]

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    for dtype, shape, tp, pp, graph_on in itertools.product(
        l_dtype_, l_shape, l_tp, l_pp, l_graph
    ):
        test_split_ar_gemma_rmsnorm(tp, pp, shape, dtype, withGraph=graph_on, distributed_init_method=init_method)
        test_fused_ar_gemma_rmsnorm(tp, pp, shape, dtype, withGraph=graph_on, distributed_init_method=init_method)
