# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public attention APIs must preserve sink logits across the SWA boundary."""

from itertools import pairwise

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha import flash_attn_func, flash_attn_varlen_func

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in ("gfx942", "gfx950"),
    reason="Regression covers gfx9 FMHA v3 dispatch",
)


def _reference(q, k, v, sink):
    heads = q.shape[1]
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float().expand(-1, heads, -1))
    scores *= q.shape[-1] ** -0.5
    pos = torch.arange(q.shape[0], device=q.device)
    mask = (pos[None, :] > pos[:, None]) | (pos[None, :] < pos[:, None] - 128)
    scores.masked_fill_(mask[None], -torch.inf)
    if sink is not None:
        scores = torch.cat([scores, sink[:, None, None].expand(-1, q.shape[0], 1)], -1)
    probs = scores.softmax(-1)[..., : k.shape[0]]
    return torch.einsum("hqk,khd->qhd", probs, v.float().expand(-1, heads, -1))


@pytest.mark.parametrize("layout", ["dense", "varlen"])
@pytest.mark.parametrize("lengths", [(1,), (54,), (128,), (129,), (54, 129)])
@pytest.mark.parametrize("qk_dim", [128, 192])
@torch.no_grad()
def test_sink_dispatch_preserves_short_attention(layout, lengths, qk_dim):
    if layout == "dense" and len(set(lengths)) > 1:
        pytest.skip("Ragged lengths require the varlen API")
    torch.manual_seed(123)
    total = sum(lengths)
    # Match MiMo TP8: 16 query heads, 1 KV head, native V128.
    q = torch.randn(total, 16, qk_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, 1, qk_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total, 1, 128, device="cuda", dtype=torch.bfloat16)
    sink = torch.linspace(-2, 5, 16, device="cuda", dtype=torch.float32)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)

    for sink_arg in (sink, None):
        kwargs = {"causal": True, "window_size": (128, 0, 0), "sink_ptr": sink_arg}
        if layout == "varlen":
            actual = flash_attn_varlen_func(
                q, k, v, cu, cu, max(lengths), max(lengths), **kwargs
            )
        else:
            actual = flash_attn_func(q[None], k[None], v[None], **kwargs)[0]
        expected = torch.cat(
            [
                _reference(q[start:end], k[start:end], v[start:end], sink_arg)
                for start, end in pairwise(offsets)
            ]
        )
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
