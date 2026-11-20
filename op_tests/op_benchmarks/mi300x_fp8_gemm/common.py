# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inputs, independent sampled references and packed row plans for MI300X GEMM."""

import csv
import json
import os
import random
from pathlib import Path

import torch

from aiter.jit.core import AITER_CONFIGS
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle, get_CKGEMM_config
from aiter.ops.shuffle import shuffle_weight

FP8 = torch.float8_e4m3fnuz
BF16 = torch.bfloat16
CONFIGS = Path(__file__).resolve().parent / "configs"
SIZES = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
PROJECTIONS = ((1536, 4096), (32768, 1536), (4096, 32768))


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


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
    def __init__(self, dims):
        self.m, self.n, self.k = m, n, k = dims
        if min(m, n, k) <= 0 or n % 16 or k % 128:
            raise ValueError(
                "Packed FNUZ inputs need positive M, 16-aligned N, and whole K128 groups"
            )
        torch.manual_seed(2026)
        self.x, self.w = make_input(m, k), make_input(n, k)
        self.packed = shuffle_weight(self.w, layout=(16, 16))
        self.out = torch.empty((m, n), device="cuda", dtype=BF16)
        self.sx = torch.rand((m, 1), device="cuda") + 0.5
        self.sw = torch.rand((1, n), device="cuda") + 0.25
        self.bx = torch.rand((m, k // 128), device="cuda") + 0.5
        self.bw = torch.rand(((n + 127) // 128, k // 128), device="cuda") + 0.25

    def reference(self, chunk_sizes=()):
        ri, ci = sample_indices(self.m), sample_indices(self.n)
        for step in chunk_sizes:
            bounds = torch.arange(step, self.m, step, device="cuda", dtype=torch.long)
            ri = torch.cat([ri, bounds - 1, bounds]).unique()
        x = self.x.view(torch.uint8)[ri].view(FP8).cpu().double()
        w = self.w.view(torch.uint8)[ci].view(FP8).cpu().double()
        ptpc = (x @ w.T) * self.sx[ri].cpu().double() * self.sw[:, ci].cpu().double()
        block = (x * self.bx[ri].cpu().double().repeat_interleave(128, dim=1)) @ (
            w * self.bw[ci // 128].cpu().double().repeat_interleave(128, dim=1)
        ).T
        return ri, ci, ptpc, block


def public_block(d, family, step):
    cfg = get_CKGEMM_config(
        step, d.n, d.k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
    )
    if not cfg or cfg["libtype"] != family:
        raise RuntimeError(f"Missing packed {family} row for {(step,d.n,d.k)}")
    if step == d.m:
        sx = packed_scales(d.bx)

        def call():
            return gemm_a8w8_blockscale_bpreshuffle(
                d.x, d.packed, sx, d.bw, BF16, out=d.out
            )

    else:
        pieces = []
        for off in range(0, d.m, step):
            sl = slice(off, off + step)
            pieces.append((d.x[sl], packed_scales(d.bx[sl]), d.out[sl]))

        def call():
            for x, sx, out in pieces:
                gemm_a8w8_blockscale_bpreshuffle(x, d.packed, sx, d.bw, BF16, out=out)
            return d.out

    return call, cfg["kernelName"]


def opus_offsets_fit(m, n, k):
    # This gfx942 kernel computes row * stride in signed int before advancing the base pointer.
    last_row_tile = ((m - 1) // 128) * 128
    return max(last_row_tile * k, last_row_tile * n) <= 2**31 - 1


def activate(table):
    # The public operator resolves its CSV in Python during graph capture.
    # Each replay then launches precisely the kernels captured under that table.
    os.environ["AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"] = str(table.resolve())
    AITER_CONFIGS.get_config_file.cache_clear()
    if (
        Path(AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE).resolve()
        != table.resolve()
    ):
        raise RuntimeError(f"Unexpected runtime table for {table}")
