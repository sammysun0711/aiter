"""Prototype native-MXFP4 MoE wrapper using the tuned MI308X dispatch.

This is intentionally small and framework-neutral.  ATOM can call the same
sequence from its ``Mxfp4MoEMethod`` after loading checkpoint tensors.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import torch


_LOCK = threading.RLock()
_DISPATCH_PATH = (
    Path(__file__).resolve().parent
    / "tuning/a16w4_triton/qwen3_8_tp8_deployment_dispatch.json"
)
_DISPATCH = json.loads(_DISPATCH_PATH.read_text())["single_layout_production_dispatch"]


@dataclass
class PreparedWeights:
    # AITER Triton consumes logical [E,K,N] column-major views.
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


def prepare_checkpoint_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> PreparedWeights:
    """Transpose the checkpoint's GGUU [E,N,K/2] tensors without requantizing.

    Expected TP8 shapes are W13=[512,512,4096], W2=[512,8192,128],
    W13-scale=[512,512,256], and W2-scale=[512,8192,8].
    """
    assert tuple(w13.shape) == (512, 512, 4096)
    assert tuple(w2.shape) == (512, 8192, 128)
    assert tuple(w13_scale.shape) == (512, 512, 256)
    assert tuple(w2_scale.shape) == (512, 8192, 8)
    return PreparedWeights(
        w13=w13.view(torch.uint8).transpose(1, 2),
        w13_scale=w13_scale.view(torch.uint8).transpose(1, 2),
        w2=w2.view(torch.uint8).transpose(1, 2),
        w2_scale=w2_scale.view(torch.uint8).transpose(1, 2),
    )


def _select_entry(token_m: int, block_m: int) -> dict:
    table = _DISPATCH
    if str(token_m) in table:
        entry = table[str(token_m)]
        if int(entry["stage1"]["block_m"]) == block_m:
            return entry
    compatible = [
        (abs(int(key) - token_m), int(key), value)
        for key, value in table.items()
        if int(value["stage1"]["block_m"]) == block_m
    ]
    if not compatible:
        raise KeyError(f"no tuned Qwen3.8 config for routing block_m={block_m}")
    return min(compatible, key=lambda item: (item[0], item[1]))[2]


def fused_moe(
    hidden_states: torch.Tensor,
    weights: PreparedWeights,
    router_logits: torch.Tensor,
) -> torch.Tensor:
    """Run the tuned single-layout TP8 expert path, including routing."""
    import aiter.ops.triton.moe.moe_op_gemm_a16w4 as a16
    from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
    from aiter.ops.triton.moe.moe_routing.routing import routing

    assert hidden_states.dtype == torch.bfloat16
    assert hidden_states.ndim == 2 and hidden_states.shape[1] == 8192
    assert router_logits.shape == (hidden_states.shape[0], 512)
    routing_data, gather_idx, scatter_idx = routing(router_logits, 10)
    entry = _select_entry(hidden_states.shape[0], routing_data.block_m)
    config_attr = (
        "get_kernel_config_triton"
        if hasattr(a16, "get_kernel_config_triton")
        else "get_kernel_config"
    )
    backend_kwargs = {"backend": "triton"} if config_attr.endswith("_triton") else {}
    original = getattr(a16, config_attr)

    def selector(m, n, k, rdata):
        return dict(entry["stage1"] if k == 8192 else entry["stage2"])

    # get_kernel_config is module-global in the current AITER API. Serialize
    # launch-time replacement; kernels themselves remain asynchronous.
    with _LOCK:
        setattr(a16, config_attr, selector)
        try:
            raw = a16.moe_gemm_a16w4(
                hidden_states,
                weights.w13,
                None,
                weights.w13_scale,
                None,
                None,
                None,
                routing_data,
                gather_indx=gather_idx,
                apply_swiglu=False,
                out_dtype=torch.bfloat16,
                **backend_kwargs,
            )
            intermediate = fused_clamp_act_mul(
                raw, activation="silu", swiglu_limit=0.0
            )
            return a16.moe_gemm_a16w4(
                intermediate,
                weights.w2,
                None,
                weights.w2_scale,
                None,
                None,
                None,
                routing_data,
                scatter_indx=scatter_idx,
                gammas=routing_data.gate_scal,
                out_dtype=torch.bfloat16,
                **backend_kwargs,
            )
        finally:
            setattr(a16, config_attr, original)
