# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared data, validation and graph timing for the customer GEMM workflow."""

import csv
import json
import random
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight

FP8 = torch.float8_e4m3fnuz
BF16 = torch.bfloat16
CONFIGS = Path(__file__).resolve().parent / "configs"


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def shape_key(row):
    return tuple(int(row[key]) for key in ("M", "N", "K"))


def require_mi300x():
    if (
        get_gfx_runtime() != "gfx942"
        or torch.cuda.get_device_properties(0).multi_processor_count != 304
    ):
        raise RuntimeError("Expected a 304-CU gfx942 MI300X on device 0")


def is_mi300x_row(row):
    return row.get("gfx", "gfx942") == "gfx942" and row["cu_num"] == "304"


def is_ptpc_row(row, packed):
    return (
        row["dtype"],
        row["outdtype"],
        row["bias"],
        row["scaleAB"],
        row["bpreshuffle"],
    ) == (str(FP8), str(BF16), "False", "True", str(packed))


def _csv_number(value):
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return value


def validate_runtime_row(shape, requested, actual, *, packed, ptpc=False):
    """Check the selected row, including FlyDSL's ID/name/layout identity."""
    keys = (
        ("libtype", "solidx", "kernelName")
        if ptpc
        else ("libtype", "kernelId", "kernelName", "splitK")
    )
    for key in keys:
        expected = requested.get(key, 0 if key == "splitK" else None)
        observed = actual.get(key, 0 if key == "splitK" else None)
        if key in ("kernelId", "solidx", "splitK"):
            expected, observed = _csv_number(expected), _csv_number(observed)
        if expected != observed:
            raise ValueError(
                f"{shape}: requested {key}={expected}, dispatch returned {observed}"
            )
    if not ptpc and actual.get("libtype") == "flydsl":
        from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_common import (
            kernel_fits_shape,
            kernels_list,
        )

        entry = kernels_list.get(_csv_number(requested["kernelId"]))
        if entry is None or (entry.name, entry.preshuffle_b) != (
            requested["kernelName"],
            packed,
        ):
            raise ValueError(
                f"{shape}: FlyDSL ID {requested['kernelId']} does not match "
                f"{requested['kernelName']} and layout in the installed registry"
            )
        if not kernel_fits_shape(entry, *shape, "gfx942"):
            raise ValueError(f"{shape}: {entry.name} does not support this shape")


def read_shapes(path):
    return list(dict.fromkeys(shape_key(row) for row in read_csv(path)))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def packed_scales(s):
    return s.T.contiguous().view_as(s)


def make_input(rows, k):
    data = torch.empty((rows, k), dtype=FP8, device="cuda")
    for start in range(0, rows, 2048):
        part = data[start : start + 2048]
        part.copy_(torch.randn(part.shape, dtype=BF16, device="cuda"))
    return data


def sample_indices(dim):
    # Check multiple tiles including the last row/column and the boundary at half-size.
    ids = {0, 1, 15, 16, 127, 128, dim // 2 - 1, dim // 2, dim - 2, dim - 1}
    ids = {i for i in ids if 0 <= i < dim}
    ids.update(random.Random(dim).sample(range(dim), min(22, dim)))
    return torch.tensor(sorted(ids), device="cuda")


def error_metric(fn, out, ri, ci, reference):
    out[ri[:, None], ci[None, :]] = float("nan")
    fn()
    return sample_error(out, ri, ci, reference)


def sample_error(out, ri, ci, reference):
    sample = out[ri[:, None], ci[None, :]].cpu().double()
    diff = sample - reference
    relative = (diff.square().mean() / reference.square().mean()).sqrt().item()
    peak = (diff.abs().max() / reference.square().mean().sqrt()).item()
    if not (relative < 0.006 and peak < 0.035):
        raise ValueError(
            f"sampled numerical check failed: relative RMS={relative}, max/RMS={peak}"
        )
    return relative, peak


class Inputs:
    def __init__(self, dims, *, ptpc=False, block_tuner=False, allocate_output=True):
        if ptpc and block_tuner:
            raise ValueError("Choose either PTPC or block-scale tuner inputs")
        self.m, self.n, self.k = m, n, k = dims
        if min(m, n, k) <= 0 or n % 16 or k % 128:
            raise ValueError(
                "Packed FNUZ inputs need positive M, 16-aligned N, and whole K128 groups"
            )
        self.seed = 0 if ptpc or block_tuner else 2026
        self.recipe = (
            "gradlib_ptpc"
            if ptpc
            else "native_blockscale" if block_tuner else "block128_signed"
        )
        self.sx = self.sw = self.bx = self.bw = None
        torch.manual_seed(self.seed)
        if ptpc:
            from aiter import pertoken_quant

            # Match gradlib.GemmTuner.generate_data: draw both tensors before
            # quantizing, including its default-dtype randn -> BF16 conversion.
            x = torch.randn((m, k), device="cuda").to(BF16)
            w = torch.randn((n, k), device="cuda").to(BF16)
            self.x, self.sx = pertoken_quant(x, quant_dtype=FP8)
            self.w, weight_scales = pertoken_quant(w, quant_dtype=FP8)
            self.sw = weight_scales.T.contiguous()
        elif block_tuner:
            # Match the native block-scale tuner's seed, draw order and scales.
            self.x = (torch.rand((m, k), dtype=torch.float16, device="cuda") / 10).to(
                FP8
            )
            self.w = (torch.rand((n, k), dtype=torch.float16, device="cuda") / 10).to(
                FP8
            )
            self.bx = torch.rand((m, k // 128), dtype=torch.float32, device="cuda")
            self.bw = torch.rand(
                ((n + 127) // 128, k // 128), dtype=torch.float32, device="cuda"
            )
        else:
            self.x, self.w = make_input(m, k), make_input(n, k)
        self.packed = shuffle_weight(self.w, layout=(16, 16))
        self.out = (
            torch.empty((m, n), device="cuda", dtype=BF16) if allocate_output else None
        )
        if not ptpc and not block_tuner:
            self.sx = torch.rand((m, 1), device="cuda") + 0.5
            self.sw = torch.rand((1, n), device="cuda") + 0.25
            self.bx = torch.rand((m, k // 128), device="cuda") + 0.5
            self.bw = torch.rand(((n + 127) // 128, k // 128), device="cuda") + 0.25

    def reference(self):
        ri, ci = sample_indices(self.m), sample_indices(self.n)
        x = self.x.view(torch.uint8)[ri].view(FP8).cpu().double()
        w = self.w.view(torch.uint8)[ci].view(FP8).cpu().double()
        ptpc = None
        if self.sx is not None:
            ptpc = (
                (x @ w.T) * self.sx[ri].cpu().double() * self.sw[:, ci].cpu().double()
            )
        block = None
        if self.bx is not None:
            block = (x * self.bx[ri].cpu().double().repeat_interleave(128, dim=1)) @ (
                w * self.bw[ci // 128].cpu().double().repeat_interleave(128, dim=1)
            ).T
        return ri, ci, ptpc, block


def opus_offsets_fit(m, n, k):
    # This gfx942 kernel computes row * stride in signed int before advancing the base pointer.
    last_row_tile = ((m - 1) // 128) * 128
    return max(last_row_tile * k, last_row_tile * n) <= 2**31 - 1


def capture_result(fn, stream, ops):
    """Warm once, then capture complete GEMM calls; return the actual output."""
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(ops):
            result = fn()
    return graph, result


def capture(fn, stream, ops):
    return capture_result(fn, stream, ops)[0]


def time_group(graph, ops):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    graph.replay()
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / ops
