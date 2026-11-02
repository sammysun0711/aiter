"""CPU API-boundary checks using exact source and mocked kernel/enum dependencies.

This intentionally avoids importing GPU-only AITER backends on a host with no
visible GPU. It does not replace numerical or graph validation of the kernels.
"""

import ast
import enum
import functools
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch


class QuantType(enum.Enum):
    per_1x32 = 0
    per_Token = 1


class ActivationType(enum.Enum):
    Silu = 0
    Relu = 1


@dataclass
class Metadata:
    stage1: object
    run_1stage: bool = False
    prequant: bool = True


@pytest.fixture
def api():
    root = Path(__file__).resolve().parents[1]
    src = root / "aiter/fused_moe.py"
    tree = ast.parse(src.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "fused_moe_a8w4_prequantized"
    )
    common = ast.parse((root / "aiter/ops/flydsl/moe_common.py").read_text())
    gate = next(
        n for n in common.body if isinstance(n, ast.ClassDef) and n.name == "GateMode"
    )
    core = Mock(return_value=object())
    ns = dict(
        torch=torch,
        get_gfx=lambda: "gfx942",
        QuantType=QuantType,
        ActivationType=ActivationType,
        Enum=enum.Enum,
        replace=replace,
        dtypes=NS(fp4x2=torch.float4_e2m1fn_x2, bf16=torch.bfloat16),
        _flydsl_stage1_wrapper=lambda: None,
        _fused_moe_impl=core,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[gate, node], type_ignores=[])),
            str(src),
            "exec",
        ),
        ns,
    )
    return ns, core


def inputs(rows=3):
    return dict(
        hidden_states=torch.ones(rows, 64, dtype=torch.float8_e4m3fnuz),
        w1=torch.empty(2, 8, 32, dtype=torch.float4_e2m1fn_x2),
        w2=torch.empty(2, 64, 2, dtype=torch.float4_e2m1fn_x2),
        topk_weight=torch.ones(rows, 2),
        topk_ids=torch.zeros(rows, 2, dtype=torch.int32),
        a1_scale=torch.arange(rows, dtype=torch.float32) + 1,
    )


def test_fp8_storage_and_scale_are_forwarded_without_requantization(api):
    ns, core = api
    args = inputs()
    out = ns["fused_moe_a8w4_prequantized"](**args)
    assert out is core.return_value
    positional, kw = core.call_args
    assert positional[0] is args["hidden_states"]
    assert positional[1] is args["w1"] and positional[2] is args["w2"]
    assert kw["dtype"] == torch.bfloat16 and kw["_q_dtype_a"] == torch.bfloat16
    scale = kw["_stage1_extra_args"]["a8w4_input_scale"]
    assert scale.data_ptr() == args["a1_scale"].data_ptr()
    stage = functools.partial(
        ns["_flydsl_stage1_wrapper"], kernelName="flydsl_moe1_afp8r_wfp4_test"
    )
    original = Metadata(stage)
    selected = kw["_metadata_transform"](original)
    assert original.prequant is True and selected.prequant is False
    assert selected.stage1 is stage


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "weights",
        "rank",
        "strided",
        "scale_dtype",
        "scale_size",
        "scale_stride",
        "ep",
        "quant_type",
        "activation",
        "gate",
        "hidden_pad",
        "inter_pad",
        "output_dtype",
        "internal_override",
        "architecture",
    ],
)
def test_invalid_input_rejected_before_core(api, case):
    ns, core = api
    args = inputs()
    if case == "dtype":
        args["hidden_states"] = args["hidden_states"].to(torch.bfloat16)
    elif case == "weights":
        args["w2"] = torch.empty(2, 64, 2, dtype=torch.bfloat16)
    elif case == "rank":
        args["hidden_states"] = args["hidden_states"].reshape(-1)
    elif case == "strided":
        args["hidden_states"] = args["hidden_states"][:, ::2]
    elif case == "scale_dtype":
        args["a1_scale"] = args["a1_scale"].to(torch.bfloat16)
    elif case == "scale_size":
        args["a1_scale"] = torch.ones(6)
    elif case == "scale_stride":
        args["a1_scale"] = torch.ones(6)[::2]
    elif case == "ep":
        args["expert_mask"] = torch.ones(2)
    elif case == "quant_type":
        args["quant_type"] = QuantType.per_Token
    elif case == "activation":
        args["activation"] = ActivationType.Relu
    elif case == "gate":
        args["gate_mode"] = "interleave"
    elif case == "hidden_pad":
        args["hidden_pad"] = 64
    elif case == "inter_pad":
        args["intermediate_pad"] = 64
    elif case == "output_dtype":
        args["dtype"] = torch.float32
    elif case == "internal_override":
        args["_metadata_transform"] = object()
    elif case == "architecture":
        ns["get_gfx"] = lambda: "gfx950"
    with pytest.raises(ValueError):
        ns["fused_moe_a8w4_prequantized"](**args)
    core.assert_not_called()


@pytest.mark.parametrize("case", ["one_stage", "wrong_wrapper", "wrong_kernel"])
def test_wrong_kernel_selection_is_rejected(api, case):
    ns, core = api
    ns["fused_moe_a8w4_prequantized"](**inputs())
    transform = core.call_args.kwargs["_metadata_transform"]
    fn = (lambda: None) if case == "wrong_wrapper" else ns["_flydsl_stage1_wrapper"]
    name = (
        "flydsl_moe1_abf16_wfp4_test"
        if case == "wrong_kernel"
        else "flydsl_moe1_afp8r_wfp4_test"
    )
    with pytest.raises(ValueError, match="selected native A8W4"):
        transform(
            Metadata(
                functools.partial(fn, kernelName=name), run_1stage=case == "one_stage"
            )
        )


def test_empty_input_returns_bf16_without_core(api):
    ns, core = api
    out = ns["fused_moe_a8w4_prequantized"](**inputs(0))
    assert out.shape == (0, 64) and out.dtype == torch.bfloat16
    core.assert_not_called()
