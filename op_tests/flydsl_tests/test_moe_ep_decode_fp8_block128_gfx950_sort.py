# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exact route preservation and dynamic graph checks for local EP sorting."""
import pytest
import torch

fc = pytest.importorskip("flydsl.compiler")
if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith(
    "gfx950"
):
    pytest.skip("The block-FP8 EP backend requires gfx950", allow_module_level=True)

from aiter.ops.flydsl.kernels.moe_ep_decode_fp8_block128_gfx950.sort import local_sort_launcher


@pytest.mark.parametrize(
    "capacity,topk,global_e,local_e,block,hidden",
    [
        (65, 8, 12, 4, 16, 1024),
        (1024, 8, 384, 48, 16, 1024),
        (96, 8, 384, 48, 32, 0),
        (1024, 1, 1024, 64, 32, 0),
    ],
)
def test_routes_mask_count_and_clear(capacity, topk, global_e, local_e, block, hidden):
    torch.manual_seed(1234)
    device = torch.device("cuda", 0)
    ids = torch.randint(global_e, (capacity, topk), device=device, dtype=torch.int32)
    weights = torch.rand(capacity, topk, device=device)
    weights[::3, -1] = 0
    mask = torch.zeros(global_e, device=device, dtype=torch.int32)
    mask[torch.randperm(global_e, device=device)[:local_e]] = 1
    count = torch.zeros(1, device=device, dtype=torch.int32)
    maximum = capacity * topk + local_e * block
    sorted_ids = torch.empty(maximum, device=device, dtype=torch.int32)
    sorted_weights = torch.empty(maximum, device=device)
    sorted_experts = torch.empty((maximum + block - 1) // block, device=device, dtype=torch.int32)
    valid = torch.empty(2, device=device, dtype=torch.int32)
    output = torch.full((capacity, hidden), 3.0, device=device, dtype=torch.bfloat16)
    args = [
        t.view(-1)
        for t in (
            ids,
            weights,
            mask,
            count,
            sorted_ids,
            sorted_weights,
            sorted_experts,
            valid,
            output,
        )
    ] + [torch.cuda.current_stream(device)]
    launcher = local_sort_launcher(capacity, topk, global_e, local_e, block, hidden, device.index)
    compiled = fc.compile[{}](
        launcher, *[fc.from_torch_tensor(t) if isinstance(t, torch.Tensor) else t for t in args]
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*args[:-1], torch.cuda.current_stream(device))

    for n in [1, min(63, capacity), capacity, 0, min(33, capacity)]:
        count.fill_(n)
        mask.copy_(mask.roll(3))
        ids.copy_((ids + 1) % global_e)
        output.fill_(3)
        graph.replay()
        torch.cuda.synchronize()

        route_ids, route_weights = ids[:n].cpu(), weights[:n].cpu()
        mapping = {g: l for l, g in enumerate(mask.cpu().nonzero().flatten().tolist())}
        expected, counts = [], [0] * local_e
        for row in range(n):
            for slot in range(topk):
                expert, weight = int(route_ids[row, slot]), float(route_weights[row, slot])
                if expert in mapping and weight != 0:
                    local = mapping[expert]
                    expected.append((local, row | (slot << 24), weight))
                    counts[local] += 1
        padded = sum((value + block - 1) // block * block for value in counts)
        assert valid.tolist() == [padded, n]
        sids, sw, se = (
            sorted_ids[:padded].cpu(),
            sorted_weights[:padded].cpu(),
            sorted_experts[: padded // block].cpu(),
        )
        actual = [
            (int(se[i // block]), int(sids[i]), float(sw[i])) for i in range(padded) if sw[i] != 0
        ]
        assert sorted(actual) == sorted(expected)
        assert sids[sw == 0].eq(capacity).all()
        assert output[:n].eq(0).all()
        assert output[n:].eq(3).all()
