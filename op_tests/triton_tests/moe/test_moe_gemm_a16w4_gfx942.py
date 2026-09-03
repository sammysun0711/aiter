"""Focused gfx942 coverage for the backported A16W4 MoE primitive.

The dimensions match one TP8 shard of Qwen3.8-2.4T-A95B, while the expert
count is reduced to keep the unit test's memory and runtime small.  Qwen's
GGUU gate/up layout is activated between the two GEMMs instead of using the
interleaved SwiGLU path inside the generic kernel.
"""

import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from aiter.ops.triton.moe.moe_op_gemm_a16w4 import (
    get_qwen3_8_tp8_kernel_config,
    moe_gemm_a16w4,
    moe_gemm_torch,
)
from aiter.ops.triton.moe.moe_routing.routing import routing
from aiter.ops.triton.moe.moe_routing.topk import topk as routing_topk
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp
from aiter.ops.triton.utils._triton.arch_info import is_moe_a16w4_avail


def test_qwen3_8_tp8_tuned_dispatch() -> None:
    required_points = {
        1: 16,
        2: 16,
        4: 16,
        8: 16,
        16: 16,
        32: 16,
        64: 16,
        128: 16,
        256: 16,
        512: 16,
        870: 16,
        871: 32,
        1024: 32,
        1689: 32,
        1690: 64,
        2048: 64,
        3327: 64,
        3328: 128,
        4096: 128,
        6144: 128,
        8192: 128,
    }
    for token_m, block_m in required_points.items():
        routing_data = SimpleNamespace(
            block_m=block_m,
            n_expts_tot=512,
            n_expts_act=10,
        )
        for n, k in ((512, 8192), (8192, 256)):
            config = get_qwen3_8_tp8_kernel_config(
                token_m * routing_data.n_expts_act,
                n,
                k,
                routing_data,
            )
            assert config is not None
            assert config["block_m"] == block_m
            assert config["block_k"] >= 128
            assert config["matrix_instr_nonkdim"] == 16

    # Unlisted shapes must still find a compatible nearest-neighbor entry in
    # each routing BLOCK_M region rather than dropping to the generic fallback.
    for token_m, block_m in ((3, 16), (900, 32), (2000, 64), (7000, 128)):
        routing_data = SimpleNamespace(
            block_m=block_m,
            n_expts_tot=512,
            n_expts_act=10,
        )
        assert (
            get_qwen3_8_tp8_kernel_config(token_m * 10, 512, 8192, routing_data)
            is not None
        )
        assert (
            get_qwen3_8_tp8_kernel_config(token_m * 10, 8192, 256, routing_data)
            is not None
        )

    prefill_stage1 = get_qwen3_8_tp8_kernel_config(
        81920,
        512,
        8192,
        SimpleNamespace(block_m=128, n_expts_tot=512, n_expts_act=10),
    )
    prefill_stage2 = get_qwen3_8_tp8_kernel_config(
        81920,
        8192,
        256,
        SimpleNamespace(block_m=128, n_expts_tot=512, n_expts_act=10),
    )
    prefill_8000_stage1 = get_qwen3_8_tp8_kernel_config(
        80000,
        512,
        8192,
        SimpleNamespace(block_m=128, n_expts_tot=512, n_expts_act=10),
    )
    prefill_8000_stage2 = get_qwen3_8_tp8_kernel_config(
        80000,
        8192,
        256,
        SimpleNamespace(block_m=128, n_expts_tot=512, n_expts_act=10),
    )

    assert prefill_stage1 == {
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
    }
    assert prefill_stage2 == {
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
    }
    assert prefill_8000_stage1 == prefill_stage1
    assert prefill_8000_stage2 == prefill_stage2
    assert (
        get_qwen3_8_tp8_kernel_config(
            40,
            512,
            8192,
            SimpleNamespace(block_m=16, n_expts_tot=16, n_expts_act=10),
        )
        is None
    )


@pytest.mark.skipif(
    not is_moe_a16w4_avail(), reason="A16W4 MoE is unsupported on this GPU"
)
def test_topk10_512_experts_matches_torch() -> None:
    torch.manual_seed(3810)
    logits = torch.randn((8, 512), device="cuda", dtype=torch.bfloat16)

    actual_weights, actual_ids, _ = routing_topk(logits, 10, apply_softmax=True)
    reference_values, _ = torch.topk(logits, 10, dim=-1)

    actual_order = torch.argsort(actual_ids, dim=-1)
    actual_ids = torch.gather(actual_ids.int(), 1, actual_order)
    actual_weights = torch.gather(actual_weights.float(), 1, actual_order)
    actual_values = torch.gather(logits.float(), 1, actual_ids.long())
    reference_threshold = reference_values.float().amin(dim=-1, keepdim=True)
    expected_weights = torch.softmax(actual_values, dim=-1)

    # BF16 logits can tie at the top-k boundary, so different valid expert ids
    # may be selected for equal values. Validate the selection threshold and
    # weights rather than imposing torch.topk's tie-breaking order.
    assert torch.all(actual_values >= reference_threshold)
    assert torch.all(actual_ids[:, 1:] != actual_ids[:, :-1])
    assert torch.allclose(actual_weights, expected_weights, rtol=1.0e-2, atol=1.0e-3)


def _assert_close(ref: torch.Tensor, actual: torch.Tensor) -> None:
    ref = ref.float()
    actual = actual.float()
    diff = (ref - actual).abs()
    scale = ref.abs().amax().clamp_min(1.0e-30)
    rel = diff / torch.maximum(ref.abs(), scale * 1.0e-6)

    assert torch.isfinite(actual).all()
    assert diff.max().item() <= 2.0e-2
    assert torch.sqrt(torch.square(rel).mean()).item() <= 4.0e-2


@pytest.mark.skipif(
    not is_moe_a16w4_avail(), reason="A16W4 MoE is unsupported on this GPU"
)
def test_qwen3_8_tp8_a16w4_two_stage() -> None:
    torch.manual_seed(38)

    m = 4
    hidden_size = 8192
    intermediate_size_per_rank = 256
    num_experts = 16
    topk = 10

    logits = torch.randn((m, num_experts), device="cuda", dtype=torch.float16)
    routing_data, gather_idx, scatter_idx = routing(logits, topk)
    hidden_states = (
        torch.randn((m, hidden_size), device="cuda", dtype=torch.bfloat16) * 0.02
    )

    # Stage 1: [E, K, 2N] -> packed [E, K/2, 2N].
    w13 = (
        torch.randn(
            (
                num_experts,
                hidden_size,
                2 * intermediate_size_per_rank,
            ),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    w13_quant, w13_scale = downcast_to_mxfp(w13, torch.uint8, axis=1)
    w13_dequant = upcast_from_mxfp(w13_quant, w13_scale, torch.bfloat16, axis=1)

    stage1_ref = moe_gemm_torch(
        hidden_states,
        w13_dequant,
        None,
        routing_data,
        gather_idx,
        None,
        None,
        False,
    )
    stage1 = moe_gemm_a16w4(
        hidden_states,
        w13_quant,
        None,
        w13_scale,
        routing_data=routing_data,
        gather_indx=gather_idx,
        out_dtype=torch.bfloat16,
    )
    _assert_close(stage1_ref, stage1)

    # Qwen checkpoint order is GGUU, not the interleaved GPT-OSS layout.
    inter_ref = (
        F.silu(stage1_ref[:, :intermediate_size_per_rank].float())
        * stage1_ref[:, intermediate_size_per_rank:].float()
    ).to(torch.bfloat16)
    inter = (
        F.silu(stage1[:, :intermediate_size_per_rank].float())
        * stage1[:, intermediate_size_per_rank:].float()
    ).to(torch.bfloat16)

    # Stage 2: [E, K, N] -> packed [E, K/2, N].
    w2 = (
        torch.randn(
            (num_experts, intermediate_size_per_rank, hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    w2_quant, w2_scale = downcast_to_mxfp(w2, torch.uint8, axis=1)
    w2_dequant = upcast_from_mxfp(w2_quant, w2_scale, torch.bfloat16, axis=1)

    output_ref = moe_gemm_torch(
        inter_ref,
        w2_dequant,
        None,
        routing_data,
        None,
        scatter_idx,
        routing_data.gate_scal,
        False,
    )
    output = moe_gemm_a16w4(
        inter,
        w2_quant,
        None,
        w2_scale,
        routing_data=routing_data,
        scatter_indx=scatter_idx,
        gammas=routing_data.gate_scal,
        out_dtype=torch.bfloat16,
    )
    _assert_close(output_ref, output)
