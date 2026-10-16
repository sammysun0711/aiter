# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import ctypes
import logging
import mmap
import os
import threading
import time
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import benchmark, checkAllclose, perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def allreduce_custom(
    tp_size,
    pp_size,
    rankID,
    x,
    withGraph=False,
    distributed_init_method: str | None = None,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            out = tensor_model_parallel_all_reduce(x)
        out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = (out, us)
    else:

        @perftest()
        def run_ca(x):
            return tensor_model_parallel_all_reduce(x)

        out = run_ca(x)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


@benchmark()
def test_allreduce_custom(
    tp_size,
    pp_size,
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: str | None = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        rets.append(
            pool.apply_async(
                allreduce_custom,
                args=(tp_size, pp_size, i, x, withGraph, distributed_init_method),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    all_us = [us for _, us in rets]
    max_err = 0.0
    for out, us in rets:
        msg = f"test_allreduce_custom: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        err = checkAllclose(ref, out.to(ref), msg=msg)
        max_err = max(max_err, err)
    return {
        "min_us": min(all_us),
        "max_us": max(all_us),
        "err": max_err,
    }


# Custom-AR size cutoff (bytes). Mirrors _DEFAULT_CAR_MAX_SIZE in
# aiter/dist/device_communicators/custom_all_reduce.py and honors the same
# AITER_CUSTOM_AR_MAX_SIZE override. Inputs at or below this run on the custom
# kernels (larger ones fall back to RCCL), so the swept size list stops here.
_DEFAULT_CAR_MAX_BYTES = 8192 * 8192


def _car_max_bytes() -> int:
    e = os.environ.get("AITER_CUSTOM_AR_MAX_SIZE", "")
    try:
        v = int(e)
        if v > 0:
            return v
    except ValueError:
        pass
    return _DEFAULT_CAR_MAX_BYTES


# Back-to-back LL128 messages of different sizes per TP size (see the LL / LL128
# dispatch in custom_all_reduce.cuh). Three calls per round alternate the bank
# each size lands in, which the overlap needs. TP8 has no LL128 range.
_LL128_MIXED_SIZES = {
    2: [16 * 7168, 16 * 7168, 8 * 7168],
    4: [4 * 8192, 4 * 8192, 28800],
}


def _start_queue_evictions(stop):
    """Repeatedly evict this process's GPU queues so its kernels pause mid-flight.

    Toggling the protection of registered host memory makes KFD preempt every
    queue of the process, which lets the peers run ahead as on a busy host.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    hip = ctypes.CDLL("libamdhip64.so")
    buf = mmap.mmap(-1, 256 << 20)
    buf.write(b"\1" * len(buf))
    view = ctypes.c_char.from_buffer(buf)
    nbytes = ctypes.c_size_t(len(buf))
    assert hip.hipHostRegister(ctypes.c_void_p(ctypes.addressof(view)), nbytes, 0) == 0

    def evict():
        # Referencing view keeps the mapping alive for the thread's lifetime.
        addr = ctypes.c_void_p(ctypes.addressof(view))
        while not stop.is_set():
            assert libc.mprotect(addr, ctypes.c_size_t(2 << 20), 1) == 0
            assert libc.mprotect(addr, ctypes.c_size_t(2 << 20), 3) == 0
            time.sleep(0.002)

    threading.Thread(target=evict, daemon=True).start()


def _allreduce_ll128_mixed_size_regression(
    rank, tp_size, sizes, iterations, distributed_init_method, stop
):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "gloo", world_size=tp_size, rank=rank, init_method=distributed_init_method
    )
    ca_comm = CustomAllreduce(dist.group.WORLD, device)
    assert not ca_comm.disabled, "The LL128 regression requires custom allreduce"
    generator = torch.Generator().manual_seed(2026)
    inputs, refs = [], []
    for n in sizes:
        x = torch.randint(-8, 9, (tp_size, n), generator=generator, dtype=torch.int16)
        inputs.append(x[rank].to(device, torch.bfloat16))
        refs.append(x.sum(dim=0, dtype=torch.float32).to(device, torch.bfloat16))
    # Opt-in: how the evictions behave depends on the host's KFD driver.
    evict = os.environ.get("AITER_TEST_LL128_EVICT") == "1"
    # Rank 1 reads the peer slots that a faster rank's next call overlaps.
    if evict and rank == 1:
        _start_queue_evictions(stop)
    dist.barrier()
    done = torch.cuda.Event()
    for iteration in range(iterations):
        outs = [ca_comm.custom_all_reduce(x, use_new=True) for x in inputs]
        if iteration % 100 != 99:
            continue
        done.record()
        deadline, evicting = time.time() + 30, evict and not stop.is_set()
        while not done.query():
            if time.time() > deadline:
                if not evicting:
                    logger.error(f"{rank=} LL128 hang before {iteration=} {sizes=}")
                    os._exit(1)
                logger.warning(f"{rank=} stalled before {iteration=}; stop evicting")
                stop.set()
                deadline, evicting = time.time() + 60, False
            time.sleep(0.001)
        for out, ref in zip(outs, refs):
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
    stop.set()
    torch.cuda.synchronize()
    dist.barrier()
    ca_comm.close()
    dist.destroy_process_group()


def test_allreduce_ll128_mixed_size_regression(tp_size, iterations=1000):
    # A rank that finished one call used to overwrite a slower peer's unread
    # LL128 slots when the next call had a different size, hanging both ranks.
    sizes = _LL128_MIXED_SIZES.get(tp_size)
    if sizes is None:
        return
    torch.multiprocessing.spawn(
        _allreduce_ll128_mixed_size_regression,
        args=(
            tp_size,
            sizes,
            iterations,
            get_distributed_init_method(get_ip(), get_open_port()),
            torch.multiprocessing.get_context("spawn").Event(),
        ),
        nprocs=tp_size,
        join=True,
    )


def gen_sizes(dtype) -> list[int]:
    """Element counts to sweep: [1024, 2048, 4096] then 7168*k / 8192*k for
    k = 1, 2, 4, 8, ... up to the largest size custom_all_reduce serves."""
    itemsize = torch.empty(0, dtype=dtype).element_size()
    max_numel = _car_max_bytes() // itemsize
    sizes = [n for n in (1024, 2048, 4096) if n <= max_numel]
    k = 1
    while True:
        added = False
        for base in (7168, 8192):
            n = base * k
            if n <= max_numel:
                sizes.append(n)
                added = True
        if not added:
            break
        k *= 2
    return sizes


l_dtype = ["bf16", "fp16", "fp32"]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    default="bf16",
    help="data type (default: bf16)",
)
parser.add_argument(
    "-t",
    "--tp-size",
    type=int,
    choices=[2, 4, 6, 8],
    default=8,
    help="number of GPUs / tensor-parallel size (default: 8)",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    choices=["graph", "eager"],
    default="graph",
    help="execution mode (default: graph)",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="single shape override, e.g. -s 128,8192 (default: swept size list)",
)


parser.add_argument(
    "--regression",
    action="store_true",
    help="run only the BF16 LL128 mixed-size correctness checks at TP2 and TP4",
)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    dtype = dtypes.d_dtypes[args.dtype]
    if args.regression and (dtype != torch.bfloat16 or args.tp_size < 2):
        parser.error("LL128 regression requires BF16 and a TP size of at least 2")
    if (args.regression or args.shape is None) and dtype == torch.bfloat16:
        # The default TP8 has no LL128 range; exercise the TP2/TP4 ranges.
        for tp in _LL128_MIXED_SIZES:
            if tp <= args.tp_size:
                test_allreduce_ll128_mixed_size_regression(tp)
    if args.regression:
        raise SystemExit(0)
    with_graph = args.mode == "graph"
    if args.shape is not None:
        l_shape = [args.shape]
    else:
        l_shape = gen_sizes(dtype)
    df = []
    for shape in l_shape:
        ret = test_allreduce_custom(
            args.tp_size,
            1,
            shape,
            dtype,
            withGraph=with_graph,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
        )
        df.append(ret)
    df = pd.DataFrame(df)
    show_cols = [
        "tp_size",
        "shape",
        "dtype",
        "withGraph",
        "min_us",
        "max_us",
        "err",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "custom allreduce summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
